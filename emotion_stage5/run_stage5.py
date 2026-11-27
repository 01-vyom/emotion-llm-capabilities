"""Run one admitted task in the locked Stage 5 crossed safety panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time
from typing import Any

from emotion_stage2.metrics import paired_delta_dicts, summarize_signals
from emotion_stage2.tasks.base import build_prompt
from emotion_stage3.io import (
    read_jsonl,
    sha256_file,
    write_csv,
    write_environment_snapshots,
    write_jsonl,
)
from emotion_stage3.run_stage3 import load_steering_spec
from emotion_stage3.steering import SteeredGenerator, TransformersConfig
from emotion_stage4.run_stage4 import (
    COMMON_SIGNAL_KEYS,
    PROMPT_CONDITIONS,
    VECTOR_STATES,
    build_stage4_arms,
)

from .tasks import (
    TASK_NAMES,
    eligible_for_signal,
    make_task,
    task_dataset_sha256,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_locks(
    *,
    panel_lock_path: Path,
    steering_lock_path: Path,
    model: str,
    task_name: str,
    vector_dir: Path,
    dataset_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    panel_lock = _load_json(panel_lock_path)
    if panel_lock.get("stage") != 5 or panel_lock.get("version") != 1:
        raise SystemExit("Invalid Stage 5 panel lock")
    if panel_lock.get("model") != model:
        raise SystemExit("Stage 5 panel lock model does not match --model")
    if task_name not in panel_lock.get("admitted_tasks", []):
        raise SystemExit(f"Task {task_name!r} has no admitted Stage 5 DV")
    task_lock = panel_lock["tasks"][task_name]
    if task_lock["dataset_sha256"] != dataset_hash:
        raise SystemExit("Stage 5 task data do not match the admission lock")
    admitted = tuple(task_lock["admitted_signals"])
    if not admitted:
        raise SystemExit(f"Task {task_name!r} has no admitted signal")

    steering_lock = _load_json(steering_lock_path)
    if steering_lock.get("version") not in {1, 2}:
        raise SystemExit("Invalid steering lock")
    if steering_lock.get("model") != model:
        raise SystemExit("Steering lock model does not match --model")
    vector_hash = sha256_file(vector_dir / "vectors.npz")
    if steering_lock.get("vectors_sha256") != vector_hash:
        raise SystemExit("Steering lock vector hash does not match")
    return panel_lock, steering_lock, admitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--steering-lock", type=Path, required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--device-map")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument(
        "--smoke-limit",
        type=int,
        help="Run the first N locked test examples without changing the data hash.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _checkpoint_rows(
    checkpoint: Path,
    *,
    item_ids: list[int],
) -> list[dict] | None:
    if not checkpoint.exists():
        return None
    rows = read_jsonl(checkpoint)
    if [int(row["item_id"]) for row in rows] != item_ids:
        return None
    if any("raw_output" not in row for row in rows):
        return None
    return rows


def main() -> None:
    args = parse_args()
    output_has_files = (
        any(args.output_dir.iterdir()) if args.output_dir.exists() else False
    )
    if output_has_files and not args.resume:
        raise SystemExit(f"Refusing to overwrite non-empty {args.output_dir}")

    task = make_task(args.task)
    full_examples, task_context = task.load("test", None, args.seed)
    dataset_hash = task_dataset_sha256(task, full_examples)
    panel_lock, steering_lock, admitted_signals = _validate_locks(
        panel_lock_path=args.panel_lock,
        steering_lock_path=args.steering_lock,
        model=args.model,
        task_name=task.name,
        vector_dir=args.vector_dir,
        dataset_hash=dataset_hash,
    )
    if args.smoke_limit is not None and args.smoke_limit <= 0:
        raise SystemExit("--smoke-limit must be positive")
    examples = (
        full_examples
        if args.smoke_limit is None
        else full_examples[: args.smoke_limit]
    )
    direction = str(steering_lock["direction"])
    policy = str(steering_lock["policy"])
    layers = tuple(int(value) for value in steering_lock["layers"])
    alpha = float(steering_lock["alpha"])
    arms, baseline_arms = build_stage4_arms(alpha)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "generation_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
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
            max_new_tokens=task.max_new_tokens(),
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
        (row["condition"], int(row["item_id"])): row for row in prompt_rows
    }
    item_ids = [example.item_id for example in examples]

    predictions = []
    arm_times = {}
    resumed_arms = []
    started = time()
    for arm in arms:
        checkpoint = checkpoint_dir / f"{arm.name}.jsonl"
        checkpoint_rows = (
            _checkpoint_rows(checkpoint, item_ids=item_ids)
            if args.resume
            else None
        )
        if checkpoint_rows is None:
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
            outputs = generator.generate(
                rendered_by_arm[arm.name],
                steering=spec,
            )
            arm_times[arm.name] = time() - arm_started
            checkpoint_rows = [
                {"item_id": example.item_id, "raw_output": raw_output}
                for example, raw_output in zip(examples, outputs, strict=True)
            ]
            write_jsonl(checkpoint, checkpoint_rows)
        else:
            arm_times[arm.name] = 0.0
            resumed_arms.append(arm.name)

        for example, raw_row in zip(examples, checkpoint_rows, strict=True):
            raw_output = str(raw_row["raw_output"])
            prompt_row = prompt_by_arm_item[(arm.name, example.item_id)]
            parsed = task.parse(raw_output, example, task_context)
            output_tokens = len(
                generator.tokenizer.encode(raw_output, add_special_tokens=False)
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
                        output_tokens >= task.max_new_tokens()
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
    write_csv(
        args.output_dir / "paired_deltas.csv",
        paired_delta_dicts(
            predictions,
            signal_keys=signal_keys,
            baselines=baseline_arms,
            primary_signals=admitted_signals,
            seed=args.seed,
            reps=args.bootstrap_reps,
        ),
    )
    write_jsonl(
        args.output_dir / "confusions.jsonl",
        task.confusion_counts(predictions),
    )
    conditional_rows = []
    for condition in sorted({row["condition"] for row in predictions}):
        condition_rows = [
            row for row in predictions if row["condition"] == condition
        ]
        for signal in task.primary_signals:
            eligible = [
                row
                for row in condition_rows
                if eligible_for_signal(task.name, signal, row)
            ]
            conditional_rows.append(
                {
                    "task": task.name,
                    "condition": condition,
                    "signal": signal,
                    "eligible_n": len(eligible),
                    "rate": (
                        sum(float(row[signal]) for row in eligible) / len(eligible)
                        if eligible
                        else 0.0
                    ),
                    "admitted": signal in admitted_signals,
                }
            )
    write_csv(args.output_dir / "conditional_primary_rates.csv", conditional_rows)

    metadata = {
        "stage": 5,
        "task": task.name,
        "model": args.model,
        "engine": "transformers",
        "design": "admission_locked_crossed_prompt_by_steering",
        "split": "test",
        "smoke_limit": args.smoke_limit,
        "confirmatory": args.smoke_limit is None,
        "num_examples": len(examples),
        "num_prompts": len(prompt_rows),
        "seed": args.seed,
        "dataset_sha256": dataset_hash,
        "arms": [arm.name for arm in arms],
        "paired_baselines": baseline_arms,
        "prompt_conditions": list(PROMPT_CONDITIONS),
        "vector_states": [state for state, _, _ in VECTOR_STATES],
        "signal_keys": signal_keys,
        "candidate_primary_signals": list(task.primary_signals),
        "primary_hypothesis_signals": list(admitted_signals),
        "not_testable_signals": panel_lock["tasks"][task.name][
            "not_testable_signals"
        ],
        "panel_lock": panel_lock,
        "panel_lock_path": str(args.panel_lock.resolve()),
        "panel_lock_sha256": sha256_file(args.panel_lock),
        "direction": direction,
        "policy": policy,
        "layers": list(layers),
        "alpha": alpha,
        "steering_lock": steering_lock,
        "steering_lock_path": str(args.steering_lock.resolve()),
        "vector_dir": str(args.vector_dir.resolve()),
        "vectors_sha256": sha256_file(args.vector_dir / "vectors.npz"),
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "device_map": args.device_map,
        "batch_size": args.batch_size,
        "max_new_tokens": task.max_new_tokens(),
        "max_rendered_prompt_tokens": max_prompt_tokens,
        "bootstrap_reps": args.bootstrap_reps,
        "task_context": task_context,
        "task_configuration": task.metadata(),
        **task.metadata(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
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
                "resumed_arms": resumed_arms,
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
                "admitted_signals": list(admitted_signals),
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
