"""Run one Stage 7 prompt-only task on the locked frontier model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

from emotion_stage2.metrics import paired_delta_dicts, summarize_signals
from emotion_stage2.tasks.base import build_prompt
from emotion_stage2.tasks.ifeval import IFEvalTask
from emotion_stage3.io import (
    sha256_file,
    write_csv,
    write_environment_snapshots,
    write_jsonl,
)
from emotion_stage4.run_stage4 import COMMON_SIGNAL_KEYS
from emotion_stage6.frontier import BedrockGenerator

from .arms import ifeval_arms, safety_arms
from .constants import (
    BASELINE_ARM,
    BOOTSTRAP_REPS,
    IFEVAL_MAX_NEW_TOKENS,
    IFEVAL_RUNTIME_AMENDMENT_SHA256,
    PREREGISTRATION_SHA256,
    SEED,
    TASK_NAMES,
)
from .io import (
    checkpoint_rows,
    load_panel_lock,
    seed_baseline_checkpoint,
    write_jsonl_atomic,
)
from .tasks import (
    eligible_for_signal,
    examples_sha256,
    locked_examples,
    make_task,
    task_dataset_sha256,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASK_NAMES, "ifeval"), required=True)
    parser.add_argument("--frontier-lock", type=Path, required=True)
    parser.add_argument("--panel-lock", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume
    ):
        raise SystemExit(f"Refusing to overwrite non-empty {args.output_dir}")
    frontier_lock = json.loads(args.frontier_lock.read_text(encoding="utf-8"))
    model = str(frontier_lock["model_id"])
    panel_lock = None
    task_lock = None
    if args.task == "ifeval":
        task = IFEvalTask()
        examples, context = task.load("test", None, args.seed)
        admitted = ("strict_prompt",)
        not_testable = ()
        arms = ifeval_arms(prompt_only=True)
        dataset_hash = None
    else:
        if args.panel_lock is None:
            raise SystemExit("Stage 7 frontier safety requires --panel-lock")
        panel_lock = load_panel_lock(args.panel_lock, model=model)
        task_lock = panel_lock["tasks"][args.task]["difficulty_lock"]
        admitted = tuple(panel_lock["tasks"][args.task]["admitted_signals"])
        not_testable = tuple(
            panel_lock["tasks"][args.task]["not_testable_signals"]
        )
        if not admitted:
            raise SystemExit(f"Stage 7 task {args.task} has no admitted DV")
        task = make_task(args.task)
        examples = locked_examples(
            args.task,
            task_lock["selected_test_item_ids"],
        )
        if examples_sha256(examples) != task_lock["selected_examples_sha256"]:
            raise SystemExit("Selected frontier example hash mismatch")
        dataset_hash = task_dataset_sha256(task, examples)
        if dataset_hash != task_lock["selected_dataset_sha256"]:
            raise SystemExit("Selected frontier dataset hash mismatch")
        context = {}
        arms = safety_arms(args.task, prompt_only=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "generation_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    prompt_rows = []
    prompts_by_arm = {}
    for arm in arms:
        prompts = []
        for example in examples:
            prompt = build_prompt(
                task,
                example,
                context,
                condition=arm.prompt_condition,
            )
            prompts.append(prompt)
            prompt_rows.append(
                {
                    "condition": arm.name,
                    "prompt_condition": arm.prompt_condition,
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
        prompts_by_arm[arm.name] = prompts
    write_jsonl(args.output_dir / "prompts.jsonl", prompt_rows)
    item_ids = [int(example.item_id) for example in examples]
    if args.task != "ifeval":
        baseline = checkpoint_dir / f"{BASELINE_ARM}.jsonl"
        if not baseline.exists():
            seed_baseline_checkpoint(
                admission_dir=Path(
                    panel_lock["tasks"][args.task]["admission_dir"]
                ),
                checkpoint=baseline,
                item_ids=item_ids,
                bedrock=True,
            )
    generator = BedrockGenerator(
        model_id=model,
        region=str(frontier_lock["region"]),
        max_tokens=(
            IFEVAL_MAX_NEW_TOKENS
            if args.task == "ifeval"
            else task.max_new_tokens()
        ),
        workers=args.workers,
    )
    prompt_by_key = {
        (row["condition"], int(row["item_id"])): row for row in prompt_rows
    }
    predictions = []
    arm_times = {}
    resumed = []
    started = time()
    for arm in arms:
        checkpoint = checkpoint_dir / f"{arm.name}.jsonl"
        rows = checkpoint_rows(checkpoint, item_ids)
        if rows is None:
            arm_started = time()
            results = generator.generate(prompts_by_arm[arm.name])
            arm_times[arm.name] = time() - arm_started
            rows = [
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
            write_jsonl_atomic(checkpoint, rows)
        else:
            arm_times[arm.name] = 0.0
            resumed.append(arm.name)
        for example, raw in zip(examples, rows, strict=True):
            raw_output = str(raw["raw_output"])
            parsed = task.parse(raw_output, example, context)
            predictions.append(
                {
                    key: value
                    for key, value in prompt_by_key[
                        (arm.name, int(example.item_id))
                    ].items()
                    if key != "prompt"
                }
                | {
                    "raw_output": raw_output,
                    "pred": parsed.pred,
                    "parse_method": parsed.parse_method,
                    "details": parsed.details,
                    **task.score(parsed, example),
                    "stop_reason": raw["stop_reason"],
                    "input_tokens": int(raw["input_tokens"]),
                    "output_tokens": int(raw["output_tokens"]),
                    "total_tokens": int(raw["total_tokens"]),
                    "latency_ms": raw["latency_ms"],
                    "output_truncated": float(
                        raw["stop_reason"] == "max_tokens"
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
            primary_signals=admitted,
            seed=args.seed,
            reps=args.bootstrap_reps,
        ),
    )
    write_jsonl(
        args.output_dir / "confusions.jsonl",
        task.confusion_counts(predictions),
    )
    if args.task != "ifeval":
        conditional = []
        for arm in arms:
            arm_rows = [
                row for row in predictions if row["condition"] == arm.name
            ]
            for signal in task.primary_signals:
                eligible = [
                    row
                    for row in arm_rows
                    if eligible_for_signal(args.task, signal, row)
                ]
                conditional.append(
                    {
                        "task": args.task,
                        "condition": arm.name,
                        "signal": signal,
                        "eligible_n": len(eligible),
                        "rate": sum(float(row[signal]) for row in eligible)
                        / len(eligible),
                        "admitted": signal in admitted,
                    }
                )
        write_csv(
            args.output_dir / "conditional_primary_rates.csv",
            conditional,
        )
    metadata = {
        "stage": 7,
        "task": args.task,
        "model": model,
        "engine": "bedrock",
        "design": (
            "full_ladder_ifeval_prompt_only"
            if args.task == "ifeval"
            else "difficulty_locked_prompt_only"
        ),
        "split": "test",
        "confirmatory": True,
        "num_examples": len(examples),
        "num_prompts": len(prompt_rows),
        "seed": args.seed,
        "dataset_sha256": dataset_hash,
        "arms": [arm.name for arm in arms],
        "paired_baselines": [BASELINE_ARM],
        "signal_keys": signal_keys,
        "primary_hypothesis_signals": list(admitted),
        "not_testable_signals": list(not_testable),
        "panel_lock": panel_lock,
        "panel_lock_path": (
            str(args.panel_lock.resolve()) if args.panel_lock else None
        ),
        "panel_lock_sha256": (
            sha256_file(args.panel_lock) if args.panel_lock else None
        ),
        "difficulty_lock": task_lock,
        "difficulty_lock_path": (
            panel_lock["tasks"][args.task]["difficulty_lock_path"]
            if panel_lock
            else None
        ),
        "difficulty_lock_sha256": (
            panel_lock["tasks"][args.task]["difficulty_lock_sha256"]
            if panel_lock
            else None
        ),
        "admission_dir": (
            panel_lock["tasks"][args.task]["admission_dir"]
            if panel_lock
            else None
        ),
        "frontier_lock": frontier_lock,
        "frontier_lock_path": str(args.frontier_lock.resolve()),
        "frontier_lock_sha256": sha256_file(args.frontier_lock),
        "direction": None,
        "policy": None,
        "layers": [],
        "vector_dir": None,
        "vectors_sha256": None,
        "dtype": None,
        "attention_implementation": None,
        "device_map": None,
        "batch_size": None,
        "workers": args.workers,
        "max_new_tokens": (
            IFEVAL_MAX_NEW_TOKENS
            if args.task == "ifeval"
            else task.max_new_tokens()
        ),
        "ifeval_runtime_amendment_sha256": (
            IFEVAL_RUNTIME_AMENDMENT_SHA256
            if args.task == "ifeval"
            else None
        ),
        "max_rendered_prompt_tokens": None,
        "bootstrap_reps": args.bootstrap_reps,
        "task_context": context,
        "task_configuration": task.metadata(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
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
                "resumed_arms": resumed,
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
                "task": args.task,
                "num_examples": len(examples),
                "num_arms": len(arms),
                "num_predictions": len(predictions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
