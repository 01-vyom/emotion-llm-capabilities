"""Run the Stage 7 open-model IFEval prompt and steering readout."""

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
from emotion_stage3.run_stage3 import load_steering_spec
from emotion_stage3.steering import SteeredGenerator, TransformersConfig
from emotion_stage4.run_stage4 import COMMON_SIGNAL_KEYS

from .arms import ifeval_arms
from .constants import (
    BASELINE_ARM,
    BOOTSTRAP_REPS,
    IFEVAL_MAX_NEW_TOKENS,
    IFEVAL_RUNTIME_AMENDMENT_SHA256,
    PREREGISTRATION_SHA256,
    SEED,
)
from .io import (
    checkpoint_rows,
    load_steering_lock,
    write_jsonl_atomic,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--steering-lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--device-map")
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
    lock = load_steering_lock(
        args.steering_lock,
        vector_dir=args.vector_dir,
        model=args.model,
    )
    task = IFEvalTask()
    examples, context = task.load("test", None, args.seed)
    arms = ifeval_arms()
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
                    "vector_state": arm.vector_state,
                    "item_id": example.item_id,
                    "vector_kind": arm.vector_kind,
                    "alpha": arm.alpha,
                    "direction": lock["direction"] if arm.vector_kind else None,
                    "policy": lock["policy"] if arm.vector_kind else None,
                    "layers": lock["layers"] if arm.vector_kind else [],
                    **task.example_metadata(example),
                    "prompt": prompt,
                }
            )
        prompts_by_arm[arm.name] = prompts
    write_jsonl(args.output_dir / "prompts.jsonl", prompt_rows)

    generator = SteeredGenerator(
        TransformersConfig(
            model=args.model,
            max_new_tokens=IFEVAL_MAX_NEW_TOKENS,
            batch_size=args.batch_size,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            device_map=args.device_map,
        )
    )
    rendered_by_arm = {
        arm.name: generator.render_chat(prompts_by_arm[arm.name])
        for arm in arms
    }
    max_prompt_tokens = max(
        len(generator.tokenizer.encode(prompt, add_special_tokens=False))
        for prompts in rendered_by_arm.values()
        for prompt in prompts
    )
    prompt_by_key = {
        (row["condition"], int(row["item_id"])): row for row in prompt_rows
    }
    item_ids = [int(example.item_id) for example in examples]
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
                    args.vector_dir,
                    pair_name=lock["direction"],
                    policy=lock["policy"],
                    layers=tuple(int(value) for value in lock["layers"]),
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
                        output_tokens >= IFEVAL_MAX_NEW_TOKENS
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
            primary_signals=["strict_prompt"],
            seed=args.seed,
            reps=args.bootstrap_reps,
        ),
    )
    write_jsonl(
        args.output_dir / "confusions.jsonl",
        task.confusion_counts(predictions),
    )
    metadata = {
        "stage": 7,
        "task": "ifeval",
        "model": args.model,
        "engine": "transformers",
        "design": "full_ladder_ifeval_prompt_plus_steering",
        "split": "test",
        "confirmatory": True,
        "num_examples": len(examples),
        "num_prompts": len(prompt_rows),
        "seed": args.seed,
        "arms": [arm.name for arm in arms],
        "paired_baselines": [BASELINE_ARM],
        "signal_keys": signal_keys,
        "primary_hypothesis_signals": ["strict_prompt"],
        "direction": lock["direction"],
        "policy": lock["policy"],
        "layers": lock["layers"],
        "steering_lock": lock,
        "steering_lock_path": str(args.steering_lock.resolve()),
        "steering_lock_sha256": sha256_file(args.steering_lock),
        "vector_dir": str(args.vector_dir.resolve()),
        "vectors_sha256": sha256_file(args.vector_dir / "vectors.npz"),
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "device_map": args.device_map,
        "batch_size": args.batch_size,
        "max_new_tokens": IFEVAL_MAX_NEW_TOKENS,
        "ifeval_runtime_amendment_sha256": (
            IFEVAL_RUNTIME_AMENDMENT_SHA256
        ),
        "max_rendered_prompt_tokens": max_prompt_tokens,
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
    write_environment_snapshots(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "num_examples": len(examples),
                "num_arms": len(arms),
                "num_predictions": len(predictions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
