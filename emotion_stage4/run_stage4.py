"""Run the preregistered Stage 4 prompt-by-steering safety panel."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

from emotion_stage2.metrics import paired_delta_dicts, summarize_signals
from emotion_stage2.tasks.base import build_prompt
from emotion_stage3.io import (
    sha256_file,
    write_csv,
    write_environment_snapshots,
    write_jsonl,
)
from emotion_stage3.run_stage3 import load_steering_spec
from emotion_stage3.steering import SteeredGenerator, TransformersConfig

from .tasks import TASK_NAMES, make_task


PROMPT_CONDITIONS = (
    "neutral_empty",
    "neutral_length_matched",
    "stage4_calm_low",
    "stage4_calm_medium",
    "stage4_calm_high",
    "stage4_desperate_low",
    "stage4_desperate_medium",
    "stage4_desperate_high",
)
VECTOR_STATES = (
    ("none", None, 0),
    ("positive", "unit", 1),
    ("negative", "unit", -1),
    ("random", "random_unit", 1),
    ("shuffled", "shuffled_unit", 1),
)
COMMON_SIGNAL_KEYS = ("output_tokens", "output_truncated")


@dataclass(frozen=True)
class Stage4Arm:
    name: str
    prompt_condition: str
    vector_state: str
    vector_kind: str | None
    alpha: float


def build_stage4_arms(alpha: float) -> tuple[list[Stage4Arm], list[str]]:
    if alpha <= 0:
        raise ValueError("Stage 4 requires a positive locked alpha")
    arms = []
    for prompt_condition in PROMPT_CONDITIONS:
        for vector_state, vector_kind, sign in VECTOR_STATES:
            arms.append(
                Stage4Arm(
                    name=f"{prompt_condition}__{vector_state}",
                    prompt_condition=prompt_condition,
                    vector_state=vector_state,
                    vector_kind=vector_kind,
                    alpha=sign * alpha,
                )
            )
    return arms, ["neutral_length_matched__none"]


def _load_and_validate_lock(
    path: Path,
    *,
    model: str,
    vector_dir: Path,
) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("version") not in {1, 2}:
        raise SystemExit("Stage 4 steering lock must have version 1 or 2")
    if lock.get("model") != model:
        raise SystemExit("Stage 4 steering lock model does not match --model")
    vector_hash = sha256_file(vector_dir / "vectors.npz")
    if lock.get("vectors_sha256") != vector_hash:
        raise SystemExit("Stage 4 steering lock vector hash does not match")
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--steering-lock", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--device-map")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--fixture-smoke-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {args.output_dir}")
    lock = _load_and_validate_lock(
        args.steering_lock,
        model=args.model,
        vector_dir=args.vector_dir,
    )
    direction = str(lock["direction"])
    policy = str(lock["policy"])
    layers = tuple(int(value) for value in lock["layers"])
    alpha = float(lock["alpha"])
    arms, baseline_arms = build_stage4_arms(alpha)
    task = make_task(args.task)
    if args.fixture_smoke_test:
        examples, task_context = task.load_fixture()
    else:
        examples, task_context = task.load(args.split, args.limit, args.seed)
    max_new_tokens = args.max_new_tokens or task.max_new_tokens()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows = []
    raw_prompts_by_arm: dict[str, list[str]] = {}
    for arm in arms:
        arm_prompts = []
        for example in examples:
            prompt = build_prompt(
                task,
                example,
                task_context,
                condition=arm.prompt_condition,
            )
            arm_prompts.append(prompt)
            prompt_rows.append(
                {
                    "condition": arm.name,
                    "prompt_condition": arm.prompt_condition,
                    "vector_state": arm.vector_state,
                    "item_id": example.item_id,
                    "vector_kind": arm.vector_kind,
                    "alpha": arm.alpha,
                    "direction": direction if arm.vector_kind else None,
                    "policy": policy if arm.vector_kind else None,
                    "layers": list(layers) if arm.vector_kind else [],
                    **task.example_metadata(example),
                    "prompt": prompt,
                }
            )
        raw_prompts_by_arm[arm.name] = arm_prompts
    write_jsonl(args.output_dir / "prompts.jsonl", prompt_rows)

    generator = SteeredGenerator(
        TransformersConfig(
            model=args.model,
            max_new_tokens=max_new_tokens,
            batch_size=args.batch_size,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            device_map=args.device_map,
        )
    )
    rendered_by_arm = {
        arm.name: generator.render_chat(raw_prompts_by_arm[arm.name])
        for arm in arms
    }
    max_prompt_tokens = max(
        (
            len(generator.tokenizer.encode(prompt, add_special_tokens=False))
            for prompts in rendered_by_arm.values()
            for prompt in prompts
        ),
        default=0,
    )

    prompt_by_arm_item = {
        (row["condition"], int(row["item_id"])): row
        for row in prompt_rows
    }
    predictions = []
    arm_times = {}
    started = time()
    for arm in arms:
        spec = None
        if arm.vector_kind is not None:
            spec = load_steering_spec(
                args.vector_dir,
                pair_name=direction,
                policy=policy,
                layers=layers,
                vector_kind=arm.vector_kind,
                alpha=arm.alpha,
            )
        arm_started = time()
        outputs = generator.generate(rendered_by_arm[arm.name], steering=spec)
        arm_times[arm.name] = time() - arm_started
        if len(outputs) != len(examples):
            raise RuntimeError(
                f"Arm {arm.name} expected {len(examples)} outputs, "
                f"received {len(outputs)}"
            )
        for example, raw_output in zip(examples, outputs, strict=True):
            prompt_row = prompt_by_arm_item[(arm.name, example.item_id)]
            parsed = task.parse(raw_output, example, task_context)
            output_tokens = len(
                generator.tokenizer.encode(
                    raw_output,
                    add_special_tokens=False,
                )
            )
            predictions.append(
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
                    **task.score(parsed, example),
                    "output_tokens": output_tokens,
                    "output_truncated": float(
                        output_tokens >= max_new_tokens
                    ),
                }
            )
    elapsed = time() - started

    signal_keys = list(task.signal_keys) + list(COMMON_SIGNAL_KEYS)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_csv(
        args.output_dir / "summary.csv",
        summarize_signals(
            predictions,
            signal_keys=signal_keys,
            seed=args.seed,
            reps=args.bootstrap_reps,
        ),
    )
    paired_rows = paired_delta_dicts(
        predictions,
        signal_keys=signal_keys,
        baselines=baseline_arms,
        primary_signals=task.primary_signals,
        seed=args.seed,
        reps=args.bootstrap_reps,
    )
    write_csv(args.output_dir / "paired_deltas.csv", paired_rows)
    write_jsonl(
        args.output_dir / "confusions.jsonl",
        task.confusion_counts(predictions),
    )

    vector_hash = sha256_file(args.vector_dir / "vectors.npz")
    metadata = {
        "stage": 4,
        "task": task.name,
        "model": args.model,
        "engine": "transformers",
        "design": "crossed_prompt_by_steering",
        "split": args.split,
        "fixture_smoke_test": args.fixture_smoke_test,
        "num_examples": len(examples),
        "num_prompts": len(prompt_rows),
        "seed": args.seed,
        "arms": [arm.name for arm in arms],
        "paired_baselines": baseline_arms,
        "prompt_conditions": list(PROMPT_CONDITIONS),
        "vector_states": [state for state, _, _ in VECTOR_STATES],
        "signal_keys": signal_keys,
        "primary_hypothesis_signals": list(task.primary_signals),
        "direction": direction,
        "policy": policy,
        "layers": list(layers),
        "alpha": alpha,
        "steering_lock": lock,
        "steering_lock_path": str(args.steering_lock.resolve()),
        "vector_dir": str(args.vector_dir.resolve()),
        "vectors_sha256": vector_hash,
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "device_map": args.device_map,
        "batch_size": args.batch_size,
        "max_new_tokens": max_new_tokens,
        "max_rendered_prompt_tokens": max_prompt_tokens,
        "bootstrap_reps": args.bootstrap_reps,
        "task_context": task_context,
        "task_configuration": task.metadata(),
        **task.metadata(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "run_stats.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "num_generations": len(predictions),
                "generations_per_second": (
                    len(predictions) / elapsed if elapsed else None
                ),
                "total_output_tokens": sum(
                    int(row["output_tokens"]) for row in predictions
                ),
                "arm_elapsed_seconds": arm_times,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_environment_snapshots(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "task": task.name,
                "num_examples": len(examples),
                "num_arms": len(arms),
                "num_predictions": len(predictions),
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
