"""Run one admission-locked Stage 6 frontier prompt-only task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

from emotion_stage2.metrics import paired_delta_dicts, summarize_signals
from emotion_stage2.tasks.base import build_prompt
from emotion_stage3.io import (
    read_jsonl,
    sha256_file,
    write_csv,
    write_environment_snapshots,
    write_jsonl,
)
from emotion_stage4.run_stage4 import COMMON_SIGNAL_KEYS, PROMPT_CONDITIONS

from .frontier import BedrockGenerator
from .run_stage6 import BASELINE_ARM
from .tasks import (
    TASK_NAMES,
    eligible_for_signal,
    make_task,
    task_dataset_sha256,
)


def _checkpoint_rows(
    path: Path,
    item_ids: list[int],
) -> list[dict] | None:
    if not path.exists():
        return None
    rows = read_jsonl(path)
    if [int(row["item_id"]) for row in rows] != item_ids:
        return None
    return rows if all("raw_output" in row for row in rows) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--frontier-lock", type=Path, required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume
    ):
        raise SystemExit(f"Refusing to overwrite non-empty {args.output_dir}")
    frontier_lock = json.loads(args.frontier_lock.read_text(encoding="utf-8"))
    panel_lock = json.loads(args.panel_lock.read_text(encoding="utf-8"))
    model = frontier_lock["model_id"]
    if panel_lock.get("stage") != 6 or panel_lock.get("model") != model:
        raise SystemExit("Invalid frontier panel lock")
    task = make_task(args.task)
    full_examples, task_context = task.load("test", None, args.seed)
    dataset_hash = task_dataset_sha256(task, full_examples)
    task_lock = panel_lock["tasks"][task.name]
    if task_lock["dataset_sha256"] != dataset_hash:
        raise SystemExit("Frontier task data do not match admission lock")
    admitted_signals = tuple(task_lock["admitted_signals"])
    if not admitted_signals:
        raise SystemExit(f"Task {task.name!r} has no admitted frontier DV")
    examples = (
        full_examples
        if args.smoke_limit is None
        else full_examples[: args.smoke_limit]
    )
    arms = [f"{condition}__none" for condition in PROMPT_CONDITIONS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "generation_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    prompt_rows = []
    prompts_by_arm = {}
    for condition in PROMPT_CONDITIONS:
        arm = f"{condition}__none"
        prompts_by_arm[arm] = []
        for example in examples:
            prompt = build_prompt(
                task,
                example,
                task_context,
                condition=condition,
            )
            prompts_by_arm[arm].append(prompt)
            prompt_rows.append(
                {
                    "condition": arm,
                    "prompt_condition": condition,
                    "vector_state": "none",
                    "item_id": example.item_id,
                    "vector_kind": None,
                    "alpha": 0.0,
                    "direction": None,
                    "policy": None,
                    "layers": [],
                    **task.example_metadata(example),
                    "prompt": prompt,
                }
            )
    write_jsonl(args.output_dir / "prompts.jsonl", prompt_rows)
    item_ids = [example.item_id for example in examples]
    admission_dir = Path(task_lock["admission_dir"])
    baseline_checkpoint = checkpoint_dir / f"{BASELINE_ARM}.jsonl"
    if (
        args.smoke_limit is None
        and not baseline_checkpoint.exists()
        and admission_dir.exists()
    ):
        admission = read_jsonl(admission_dir / "predictions.jsonl")
        by_item = {int(row["item_id"]): row for row in admission}
        write_jsonl(
            baseline_checkpoint,
            [
                {
                    "item_id": item_id,
                    "raw_output": by_item[item_id]["raw_output"],
                    "stop_reason": by_item[item_id]["stop_reason"],
                    "input_tokens": by_item[item_id]["input_tokens"],
                    "output_tokens": by_item[item_id]["output_tokens"],
                    "total_tokens": by_item[item_id]["total_tokens"],
                    "latency_ms": by_item[item_id]["latency_ms"],
                    "reused_from_admission": True,
                }
                for item_id in item_ids
            ],
        )
    generator = BedrockGenerator(
        model_id=model,
        region=frontier_lock["region"],
        max_tokens=args.max_tokens,
        workers=args.workers,
    )
    prompt_by_key = {
        (row["condition"], int(row["item_id"])): row for row in prompt_rows
    }
    predictions = []
    arm_times = {}
    resumed_arms = []
    started = time()
    for arm in arms:
        checkpoint = checkpoint_dir / f"{arm}.jsonl"
        checkpoint_rows = _checkpoint_rows(checkpoint, item_ids)
        if checkpoint_rows is None:
            arm_started = time()
            results = generator.generate(prompts_by_arm[arm])
            arm_times[arm] = time() - arm_started
            checkpoint_rows = [
                {
                    "item_id": example.item_id,
                    "raw_output": result.raw_output,
                    "stop_reason": result.stop_reason,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                    "latency_ms": result.latency_ms,
                }
                for example, result in zip(examples, results, strict=True)
            ]
            write_jsonl(checkpoint, checkpoint_rows)
        else:
            arm_times[arm] = 0.0
            resumed_arms.append(arm)
        for example, raw_row in zip(examples, checkpoint_rows, strict=True):
            raw_output = str(raw_row["raw_output"])
            parsed = task.parse(raw_output, example, task_context)
            predictions.append(
                {
                    key: value
                    for key, value in prompt_by_key[
                        (arm, example.item_id)
                    ].items()
                    if key != "prompt"
                }
                | {
                    "raw_output": raw_output,
                    "pred": parsed.pred,
                    "parse_method": parsed.parse_method,
                    "details": parsed.details,
                    **task.score(parsed, example),
                    "input_tokens": int(raw_row["input_tokens"]),
                    "output_tokens": int(raw_row["output_tokens"]),
                    "total_tokens": int(raw_row["total_tokens"]),
                    "latency_ms": raw_row["latency_ms"],
                    "stop_reason": raw_row["stop_reason"],
                    "output_truncated": float(
                        raw_row["stop_reason"] == "max_tokens"
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
            baselines=[BASELINE_ARM],
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
    for arm in arms:
        arm_rows = [row for row in predictions if row["condition"] == arm]
        for signal in task.primary_signals:
            eligible = [
                row
                for row in arm_rows
                if eligible_for_signal(task.name, signal, row)
            ]
            conditional_rows.append(
                {
                    "task": task.name,
                    "condition": arm,
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
        "stage": 6,
        "task": task.name,
        "model": model,
        "engine": "bedrock",
        "design": "admission_locked_prompt_only",
        "split": "test",
        "smoke_limit": args.smoke_limit,
        "confirmatory": args.smoke_limit is None,
        "num_examples": len(examples),
        "num_prompts": len(prompt_rows),
        "seed": args.seed,
        "dataset_sha256": dataset_hash,
        "arms": arms,
        "paired_baselines": [BASELINE_ARM],
        "prompt_conditions": list(PROMPT_CONDITIONS),
        "dose_grid": [],
        "signal_keys": signal_keys,
        "primary_hypothesis_signals": list(admitted_signals),
        "not_testable_signals": task_lock["not_testable_signals"],
        "panel_lock": panel_lock,
        "panel_lock_path": str(args.panel_lock.resolve()),
        "panel_lock_sha256": sha256_file(args.panel_lock),
        "admission_dir": str(admission_dir),
        "frontier_lock": frontier_lock,
        "frontier_lock_path": str(args.frontier_lock.resolve()),
        "frontier_lock_sha256": sha256_file(args.frontier_lock),
        "direction": None,
        "policy": None,
        "layers": [],
        "steering_lock": None,
        "steering_lock_path": None,
        "vector_dir": None,
        "vectors_sha256": None,
        "dtype": None,
        "attention_implementation": None,
        "device_map": None,
        "batch_size": None,
        "workers": args.workers,
        "max_new_tokens": args.max_tokens,
        "max_rendered_prompt_tokens": None,
        "bootstrap_reps": args.bootstrap_reps,
        "task_context": task_context,
        "task_configuration": task.metadata(),
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
                "total_input_tokens": sum(
                    int(row["input_tokens"]) for row in predictions
                ),
                "total_output_tokens": sum(
                    int(row["output_tokens"]) for row in predictions
                ),
                "total_tokens": sum(
                    int(row["total_tokens"]) for row in predictions
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
                "model": model,
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
