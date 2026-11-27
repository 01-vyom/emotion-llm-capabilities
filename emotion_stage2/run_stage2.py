"""Run Stage 2 mechanism-matched affective prompting experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from time import time
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from emotion_stage1.affect import CONDITIONS, DEFAULT_CONDITIONS, get_condition_cues
from emotion_stage1.generation import (
    GenerationConfig,
    make_generator,
    render_chat_prompts,
)

from .metrics import paired_delta_dicts, summarize_signals
from .tasks.base import build_prompt, variant_arm
from .tasks.registry import TASK_NAMES, make_task


DEFAULT_MODEL = "Qwen/Qwen3-8B"
BASELINE_CONDITIONS = ("neutral_empty", "neutral_length_matched")
COMMON_SIGNAL_KEYS = ("output_tokens", "output_truncated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--engine", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--enable-yarn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable YaRN through vLLM hf_overrides.",
    )
    parser.add_argument(
        "--yarn-factor",
        type=float,
        help="YaRN scale factor; defaults to max_model_len / original length.",
    )
    parser.add_argument(
        "--yarn-original-max-position-embeddings",
        type=int,
        default=40960,
    )
    parser.add_argument("--prompt-format", choices=("chat", "plain"), default="chat")
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help=f"Affect conditions. Known: {', '.join(sorted(CONDITIONS))}",
    )
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument(
        "--all-variants",
        action="store_true",
        help="Run every prompt variant and label each condition arm with ::vN.",
    )
    variants.add_argument(
        "--prompt-lock",
        type=Path,
        help="JSON prompt lock produced by select_prompt_variants.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run-prompts", action="store_true")
    parser.add_argument("--fixture-smoke-test", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument(
        "--ruler-context-lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384],
    )
    parser.add_argument(
        "--ruler-depths",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 0.9],
    )
    parser.add_argument(
        "--ruler-modes",
        nargs="+",
        choices=("single", "multi", "aggregate"),
        default=["single", "multi"],
    )
    parser.add_argument("--ruler-samples-per-cell", type=int, default=4)
    parser.add_argument(
        "--ruler-hard",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ruler-distractors", type=int)
    parser.add_argument("--ruler-multi-key-count", type=int)
    parser.add_argument("--ruler-aggregate-match-count", type=int)
    parser.add_argument("--ruler-context-reserve-tokens", type=int)
    return parser.parse_args()


def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").lower()


def _default_output_dir(args: argparse.Namespace) -> Path:
    suffix = "variants" if args.all_variants else "locked"
    return Path(
        f"runs/stage2_{args.task}_{_model_slug(args.model)}_{args.split}_{suffix}"
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_prompt_lock(
    path: Path | None,
    task_name: str,
    model: str,
    task_configuration: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any] | None]:
    if path is None:
        return {}, None
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("task") != task_name:
        raise SystemExit(
            f"Prompt lock task {lock.get('task')!r} does not match {task_name!r}"
        )
    if lock.get("model") != model:
        raise SystemExit(
            f"Prompt lock model {lock.get('model')!r} does not match {model!r}"
        )
    locked_task_configuration = lock.get("task_configuration")
    if (
        locked_task_configuration is not None
        and locked_task_configuration != task_configuration
    ):
        raise SystemExit(
            "Prompt lock task configuration does not match the current run. "
            f"Locked: {locked_task_configuration!r}; "
            f"current: {task_configuration!r}"
        )
    selected = {
        condition: int(index)
        for condition, index in lock["selected_variants"].items()
    }
    return selected, lock


def _condition_variants(
    conditions: list[str],
    all_variants: bool,
    selected_variants: dict[str, int],
) -> list[tuple[str, int, str]]:
    arms = []
    for condition in conditions:
        if all_variants:
            indices = range(len(get_condition_cues(condition)))
        else:
            indices = [selected_variants.get(condition, 0)]
        for variant_index in indices:
            get_condition_cues(condition)[variant_index]
            arms.append(
                (
                    condition,
                    variant_index,
                    variant_arm(condition, variant_index, all_variants),
                )
            )
    return arms


def _write_environment_snapshots(output_dir: Path) -> None:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    (output_dir / "requirements.freeze.txt").write_text(
        freeze.stdout,
        encoding="utf-8",
    )
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "-q"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    (output_dir / "nvidia-smi.txt").write_text(gpu.stdout, encoding="utf-8")


def _output_token_counts(raw_outputs: list[str], model: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return [
        len(tokenizer.encode(output, add_special_tokens=False))
        for output in raw_outputs
    ]


def _prompt_token_counts(prompts: list[str], model: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    ]


def _validate_rendered_prompt_lengths(
    prompt_token_counts: list[int],
    max_new_tokens: int,
    max_model_len: int | None,
) -> int:
    max_prompt_tokens = max(prompt_token_counts, default=0)
    if (
        max_model_len is not None
        and max_prompt_tokens + max_new_tokens > max_model_len
    ):
        raise SystemExit(
            "Rendered prompt exceeds the configured model window: "
            f"{max_prompt_tokens} prompt tokens + {max_new_tokens} generation "
            f"tokens > max_model_len {max_model_len}. Increase the RULER "
            "context reserve or max_model_len."
        )
    return max_prompt_tokens


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.max_model_len is not None and args.max_model_len <= 0:
        raise SystemExit("--max-model-len must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    if args.tensor_parallel_size <= 0:
        raise SystemExit("--tensor-parallel-size must be positive")
    if args.enable_yarn:
        if args.engine != "vllm":
            raise SystemExit("--enable-yarn requires --engine vllm")
        if args.max_model_len is None:
            raise SystemExit("--enable-yarn requires --max-model-len")
        if args.yarn_original_max_position_embeddings <= 0:
            raise SystemExit(
                "--yarn-original-max-position-embeddings must be positive"
            )
    elif args.yarn_factor is not None:
        raise SystemExit("--yarn-factor requires --enable-yarn")


def _rope_scaling(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.enable_yarn:
        return None
    factor = args.yarn_factor
    if factor is None:
        factor = (
            args.max_model_len
            / args.yarn_original_max_position_embeddings
        )
    if factor <= 1.0:
        raise SystemExit("YaRN factor must be greater than 1")
    return {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": (
            args.yarn_original_max_position_embeddings
        ),
    }


def _check_output_dir(output_dir: Path, overwrite: bool) -> None:
    protected = (
        "predictions.jsonl",
        "summary.csv",
        "paired_deltas.csv",
        "run_stats.json",
    )
    existing = [name for name in protected if (output_dir / name).exists()]
    if existing and not overwrite:
        raise SystemExit(
            f"Output directory {output_dir} already has completed artifacts: "
            f"{', '.join(existing)}. Use a new directory or --overwrite."
        )


def main() -> None:
    args = parse_args()
    _validate_runtime_args(args)
    rope_scaling = _rope_scaling(args)
    unknown_conditions = sorted(set(args.conditions) - set(CONDITIONS))
    if unknown_conditions:
        raise SystemExit(f"Unknown conditions: {', '.join(unknown_conditions)}")
    if args.split == "test" and args.all_variants:
        raise SystemExit(
            "Do not run all prompt variants on the held-out test split. "
            "Use --prompt-lock."
        )
    if (
        args.split == "test"
        and args.prompt_lock is None
        and not args.fixture_smoke_test
    ):
        raise SystemExit(
            "Held-out test runs require --prompt-lock from a matching dev run."
        )

    output_dir = args.output_dir or _default_output_dir(args)
    _check_output_dir(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    task = make_task(args.task, args)
    task_configuration = task.metadata()
    selected_variants, prompt_lock = _load_prompt_lock(
        args.prompt_lock,
        task.name,
        args.model,
        task_configuration,
    )
    if args.prompt_lock:
        missing = sorted(set(args.conditions) - set(selected_variants))
        if missing:
            raise SystemExit(
                f"Prompt lock is missing conditions: {', '.join(missing)}"
            )

    if args.fixture_smoke_test:
        examples, task_context = task.load_fixture()
    else:
        examples, task_context = task.load(
            split=args.split,
            limit=args.limit,
            seed=args.seed,
        )
    arms = _condition_variants(
        args.conditions,
        all_variants=args.all_variants,
        selected_variants=selected_variants,
    )
    max_new_tokens = args.max_new_tokens or task.max_new_tokens()

    prompt_rows = []
    prompts = []
    for base_condition, variant_index, arm in arms:
        for example in examples:
            prompt = build_prompt(
                task,
                example,
                task_context,
                condition=base_condition,
                variant_index=variant_index,
            )
            prompt_rows.append(
                {
                    "condition": arm,
                    "base_condition": base_condition,
                    "variant_index": variant_index,
                    "item_id": example.item_id,
                    **task.example_metadata(example),
                    "prompt": prompt,
                }
            )
            prompts.append(prompt)

    model_prompts = prompts
    max_rendered_prompt_tokens = None
    if not args.dry_run_prompts:
        if args.prompt_format == "chat":
            model_prompts = render_chat_prompts(
                prompts,
                model=args.model,
                disable_thinking=args.disable_thinking,
            )
        prompt_token_counts = _prompt_token_counts(model_prompts, args.model)
        max_rendered_prompt_tokens = _validate_rendered_prompt_lengths(
            prompt_token_counts,
            max_new_tokens=max_new_tokens,
            max_model_len=args.max_model_len,
        )

    write_jsonl(output_dir / "prompts.jsonl", prompt_rows)
    baseline_arms = [
        variant_arm(condition, 0, args.all_variants)
        for condition in BASELINE_CONDITIONS
        if condition in args.conditions
    ]
    metadata = {
        "stage": 2,
        "task": task.name,
        "model": args.model,
        "engine": args.engine,
        "split": args.split,
        "fixture_smoke_test": args.fixture_smoke_test,
        "limit": args.limit,
        "num_examples": len(examples),
        "num_prompts": len(prompt_rows),
        "seed": args.seed,
        "conditions": args.conditions,
        "arms": [arm for _, _, arm in arms],
        "all_variants": args.all_variants,
        "prompt_lock": prompt_lock,
        "paired_baselines": baseline_arms,
        "signal_keys": list(task.signal_keys) + list(COMMON_SIGNAL_KEYS),
        "primary_hypothesis_signals": list(task.primary_signals),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "dtype": args.dtype,
        "max_new_tokens": max_new_tokens,
        "max_model_len": args.max_model_len,
        "max_rendered_prompt_tokens": max_rendered_prompt_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "tensor_parallel_size": args.tensor_parallel_size,
        "yarn_enabled": args.enable_yarn,
        "rope_scaling": rope_scaling,
        "vllm_allow_long_max_model_len": bool(rope_scaling),
        "long_context_note": (
            "Contexts beyond the native model window use YaRN and may reflect "
            "rope-scaling effects as well as task difficulty."
            if args.enable_yarn
            else None
        ),
        "batch_size": args.batch_size,
        "prompt_format": args.prompt_format,
        "disable_thinking": args.disable_thinking,
        "bootstrap_reps": args.bootstrap_reps,
        "task_context": task_context,
        "task_configuration": task_configuration,
        **task_configuration,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_environment_snapshots(output_dir)

    if args.dry_run_prompts:
        print(f"Wrote {len(prompt_rows)} prompts to {output_dir / 'prompts.jsonl'}")
        return

    generator = make_generator(
        args.engine,
        GenerationConfig(
            model=args.model,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            tensor_parallel_size=args.tensor_parallel_size,
            rope_scaling=rope_scaling,
        ),
    )

    start_time = time()
    raw_outputs = []
    for start in tqdm(
        range(0, len(model_prompts), args.batch_size),
        desc="generating",
    ):
        batch = model_prompts[start : start + args.batch_size]
        raw_outputs.extend(generator.generate(batch))
    elapsed_seconds = time() - start_time
    if len(raw_outputs) != len(prompt_rows):
        raise RuntimeError(
            f"Expected {len(prompt_rows)} outputs, got {len(raw_outputs)}"
        )
    output_tokens = _output_token_counts(raw_outputs, args.model)

    example_by_item = {example.item_id: example for example in examples}
    rows = []
    for prompt_row, raw_output, token_count in zip(
        prompt_rows,
        raw_outputs,
        output_tokens,
        strict=True,
    ):
        example = example_by_item[int(prompt_row["item_id"])]
        parsed = task.parse(raw_output, example, task_context)
        score = task.score(parsed, example)
        rows.append(
            {
                key: value
                for key, value in prompt_row.items()
                if key != "prompt"
            }
            | {
                "raw_output": raw_output,
                "pred": parsed.pred,
                "parse_method": parsed.parse_method,
                "details": parsed.details,
                **score,
                "output_tokens": token_count,
                "output_truncated": float(token_count >= max_new_tokens),
            }
        )

    write_jsonl(output_dir / "predictions.jsonl", rows)
    signal_keys = list(task.signal_keys) + list(COMMON_SIGNAL_KEYS)
    summary_rows = summarize_signals(
        rows,
        signal_keys=signal_keys,
        seed=args.seed,
        reps=args.bootstrap_reps,
    )
    write_csv(output_dir / "summary.csv", summary_rows)
    paired_rows = paired_delta_dicts(
        rows,
        signal_keys=signal_keys,
        baselines=baseline_arms,
        primary_signals=task.primary_signals,
        seed=args.seed,
        reps=args.bootstrap_reps,
    )
    write_csv(output_dir / "paired_deltas.csv", paired_rows)
    write_jsonl(output_dir / "confusions.jsonl", task.confusion_counts(rows))

    run_stats = {
        "elapsed_seconds": elapsed_seconds,
        "num_generations": len(rows),
        "generations_per_second": (
            len(rows) / elapsed_seconds if elapsed_seconds else None
        ),
        "total_output_tokens": sum(output_tokens),
        "output_tokens_per_second": (
            sum(output_tokens) / elapsed_seconds if elapsed_seconds else None
        ),
    }
    (output_dir / "run_stats.json").write_text(
        json.dumps(run_stats, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote predictions to {output_dir / 'predictions.jsonl'}")
    print(f"Wrote summary to {output_dir / 'summary.csv'}")
    print(f"Wrote paired deltas to {output_dir / 'paired_deltas.csv'}")


if __name__ == "__main__":
    main()
