"""Run one open model's difficulty-locked Stage 7 safety tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

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
from emotion_stage4.run_stage4 import COMMON_SIGNAL_KEYS

from .arms import safety_arms
from .constants import (
    BASELINE_ARM,
    BOOTSTRAP_REPS,
    PREREGISTRATION_SHA256,
    SEED,
    TASK_NAMES,
)
from .io import (
    checkpoint_rows,
    load_panel_lock,
    load_steering_lock,
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--steering-lock", type=Path, required=True)
    parser.add_argument("--sycophancy-output-dir", type=Path, required=True)
    parser.add_argument("--refusal-output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--device-map")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _run_task(
    *,
    task_name: str,
    output_dir: Path,
    generator: SteeredGenerator,
    panel_lock: dict,
    panel_lock_path: Path,
    steering_lock: dict,
    steering_lock_path: Path,
    vector_dir: Path,
    seed: int,
    reps: int,
    resume: bool,
) -> dict:
    task_lock = panel_lock["tasks"][task_name]["difficulty_lock"]
    admitted = tuple(panel_lock["tasks"][task_name]["admitted_signals"])
    if not admitted:
        return {
            "task": task_name,
            "status": "not_testable",
            "not_testable_signals": panel_lock["tasks"][task_name][
                "not_testable_signals"
            ],
        }
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError(f"Refusing to overwrite non-empty {output_dir}")
    examples = locked_examples(task_name, task_lock["selected_test_item_ids"])
    task = make_task(task_name)
    if examples_sha256(examples) != task_lock["selected_examples_sha256"]:
        raise ValueError(f"Selected example hash mismatch for {task_name}")
    dataset_hash = task_dataset_sha256(task, examples)
    if dataset_hash != task_lock["selected_dataset_sha256"]:
        raise ValueError(f"Selected dataset hash mismatch for {task_name}")
    context = {}
    arms = safety_arms(task_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "generation_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    prompt_rows = []
    raw_prompts_by_arm = {}
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
                    "vector_state": arm.vector_state,
                    "item_id": example.item_id,
                    "vector_kind": arm.vector_kind,
                    "alpha": arm.alpha,
                    "direction": (
                        steering_lock["direction"] if arm.vector_kind else None
                    ),
                    "policy": (
                        steering_lock["policy"] if arm.vector_kind else None
                    ),
                    "layers": (
                        steering_lock["layers"] if arm.vector_kind else []
                    ),
                    **task.example_metadata(example),
                    "prompt": prompt,
                }
            )
        raw_prompts_by_arm[arm.name] = prompts
    write_jsonl(output_dir / "prompts.jsonl", prompt_rows)
    rendered_by_arm = {
        arm.name: generator.render_chat(raw_prompts_by_arm[arm.name])
        for arm in arms
    }
    max_prompt_tokens = max(
        len(generator.tokenizer.encode(prompt, add_special_tokens=False))
        for prompts in rendered_by_arm.values()
        for prompt in prompts
    )
    item_ids = [int(example.item_id) for example in examples]
    baseline_checkpoint = checkpoint_dir / f"{BASELINE_ARM}.jsonl"
    if not baseline_checkpoint.exists():
        seed_baseline_checkpoint(
            admission_dir=Path(panel_lock["tasks"][task_name]["admission_dir"]),
            checkpoint=baseline_checkpoint,
            item_ids=item_ids,
            bedrock=False,
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
            spec = None
            if arm.vector_kind is not None:
                spec = load_steering_spec(
                    vector_dir,
                    pair_name=steering_lock["direction"],
                    policy=steering_lock["policy"],
                    layers=tuple(int(value) for value in steering_lock["layers"]),
                    vector_kind=arm.vector_kind,
                    alpha=arm.alpha,
                )
            arm_started = time()
            outputs = generator.generate(
                rendered_by_arm[arm.name],
                steering=spec,
            )
            arm_times[arm.name] = time() - arm_started
            rows = [
                {"item_id": example.item_id, "raw_output": raw_output}
                for example, raw_output in zip(examples, outputs, strict=True)
            ]
            write_jsonl_atomic(checkpoint, rows)
        else:
            arm_times[arm.name] = 0.0
            resumed.append(arm.name)
        for example, raw in zip(examples, rows, strict=True):
            raw_output = str(raw["raw_output"])
            parsed = task.parse(raw_output, example, context)
            output_tokens = len(
                generator.tokenizer.encode(
                    raw_output,
                    add_special_tokens=False,
                )
            )
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
                    "output_tokens": output_tokens,
                    "output_truncated": float(
                        output_tokens >= task.max_new_tokens()
                    ),
                }
            )
    elapsed = time() - started
    signal_keys = list(task.signal_keys) + list(COMMON_SIGNAL_KEYS)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_csv(
        output_dir / "summary.csv",
        summarize_signals(
            predictions,
            signal_keys=signal_keys,
            seed=seed,
            reps=reps,
        ),
    )
    write_csv(
        output_dir / "paired_deltas.csv",
        paired_delta_dicts(
            predictions,
            signal_keys=signal_keys,
            baselines=[BASELINE_ARM],
            primary_signals=admitted,
            seed=seed,
            reps=reps,
        ),
    )
    write_jsonl(output_dir / "confusions.jsonl", task.confusion_counts(predictions))
    conditional = []
    for arm in arms:
        arm_rows = [row for row in predictions if row["condition"] == arm.name]
        for signal in task.primary_signals:
            eligible = [
                row
                for row in arm_rows
                if eligible_for_signal(task_name, signal, row)
            ]
            conditional.append(
                {
                    "task": task_name,
                    "condition": arm.name,
                    "signal": signal,
                    "eligible_n": len(eligible),
                    "rate": sum(float(row[signal]) for row in eligible)
                    / len(eligible),
                    "admitted": signal in admitted,
                }
            )
    write_csv(output_dir / "conditional_primary_rates.csv", conditional)
    metadata = {
        "stage": 7,
        "task": task_name,
        "model": panel_lock["model"],
        "engine": "transformers",
        "design": (
            "difficulty_locked_prompt_only"
            if task_name == "sycophancy"
            else "difficulty_locked_prompt_plus_dose_sign"
        ),
        "split": "test",
        "confirmatory": True,
        "num_examples": len(examples),
        "num_prompts": len(prompt_rows),
        "seed": seed,
        "dataset_sha256": dataset_hash,
        "arms": [arm.name for arm in arms],
        "paired_baselines": [BASELINE_ARM],
        "signal_keys": signal_keys,
        "primary_hypothesis_signals": list(admitted),
        "not_testable_signals": panel_lock["tasks"][task_name][
            "not_testable_signals"
        ],
        "panel_lock": panel_lock,
        "panel_lock_path": str(panel_lock_path.resolve()),
        "panel_lock_sha256": sha256_file(panel_lock_path),
        "difficulty_lock": task_lock,
        "difficulty_lock_path": panel_lock["tasks"][task_name][
            "difficulty_lock_path"
        ],
        "difficulty_lock_sha256": panel_lock["tasks"][task_name][
            "difficulty_lock_sha256"
        ],
        "admission_dir": panel_lock["tasks"][task_name]["admission_dir"],
        "direction": (
            steering_lock["direction"] if task_name == "refusal_drift" else None
        ),
        "policy": (
            steering_lock["policy"] if task_name == "refusal_drift" else None
        ),
        "layers": (
            steering_lock["layers"] if task_name == "refusal_drift" else []
        ),
        "steering_lock": (
            steering_lock if task_name == "refusal_drift" else None
        ),
        "steering_lock_path": (
            str(steering_lock_path.resolve())
            if task_name == "refusal_drift"
            else None
        ),
        "vector_dir": (
            str(vector_dir.resolve())
            if task_name == "refusal_drift"
            else None
        ),
        "vectors_sha256": (
            sha256_file(vector_dir / "vectors.npz")
            if task_name == "refusal_drift"
            else None
        ),
        "dtype": generator.config.dtype,
        "attention_implementation": generator.config.attention_implementation,
        "device_map": generator.config.device_map,
        "batch_size": generator.config.batch_size,
        "max_new_tokens": task.max_new_tokens(),
        "max_rendered_prompt_tokens": max_prompt_tokens,
        "bootstrap_reps": reps,
        "task_context": context,
        "task_configuration": task.metadata(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (output_dir / "run_stats.json").write_text(
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
                "resumed_arms": resumed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_environment_snapshots(output_dir)
    return {
        "task": task_name,
        "status": "complete",
        "num_examples": len(examples),
        "num_arms": len(arms),
        "num_predictions": len(predictions),
    }


def main() -> None:
    args = _parse_args()
    panel_lock = load_panel_lock(args.panel_lock, model=args.model)
    steering_lock = load_steering_lock(
        args.steering_lock,
        vector_dir=args.vector_dir,
        model=args.model,
    )
    generator = SteeredGenerator(
        TransformersConfig(
            model=args.model,
            max_new_tokens=8,
            batch_size=args.batch_size,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            device_map=args.device_map,
        )
    )
    outputs = {
        "sycophancy": args.sycophancy_output_dir,
        "refusal_drift": args.refusal_output_dir,
    }
    results = [
        _run_task(
            task_name=task_name,
            output_dir=outputs[task_name],
            generator=generator,
            panel_lock=panel_lock,
            panel_lock_path=args.panel_lock,
            steering_lock=steering_lock,
            steering_lock_path=args.steering_lock,
            vector_dir=args.vector_dir,
            seed=args.seed,
            reps=args.bootstrap_reps,
            resume=args.resume,
        )
        for task_name in TASK_NAMES
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

