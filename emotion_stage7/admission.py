"""Run locked held-out Stage 7 no-steer admission for one model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

from emotion_stage2.tasks.base import build_prompt
from emotion_stage3.io import (
    read_jsonl,
    sha256_file,
    write_csv,
    write_environment_snapshots,
    write_jsonl,
)
from emotion_stage3.steering import SteeredGenerator, TransformersConfig
from emotion_stage6.frontier import BedrockGenerator

from .constants import PREREGISTRATION_SHA256, SEED, TASK_NAMES
from .tasks import (
    eligible_for_signal,
    examples_sha256,
    locked_examples,
    make_task,
    task_dataset_sha256,
)


CONDITION = "neutral_length_matched"


def task_requires_held_out_admission(lock: dict) -> bool:
    return any(
        signal_lock["dev_status"] == "calibrated"
        for signal_lock in lock["signals"].values()
    )


def _checkpoint_rows(path: Path, item_ids: list[int]) -> list[dict] | None:
    if not path.exists():
        return None
    rows = read_jsonl(path)
    if [int(row["item_id"]) for row in rows] != item_ids:
        return None
    return rows if all("raw_output" in row for row in rows) else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("transformers", "bedrock"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--frontier-lock", type=Path)
    parser.add_argument("--task-locks", nargs="+", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--device-map")
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
    frontier_lock = None
    if args.engine == "bedrock":
        if args.frontier_lock is None:
            raise SystemExit("Bedrock admission requires --frontier-lock")
        frontier_lock = json.loads(args.frontier_lock.read_text(encoding="utf-8"))
        model = str(frontier_lock["model_id"])
    else:
        if not args.model:
            raise SystemExit("Transformers admission requires --model")
        model = args.model

    locks = {}
    for path in args.task_locks:
        lock = json.loads(path.read_text(encoding="utf-8"))
        if (
            lock.get("stage") != 7
            or lock.get("artifact") != "difficulty_lock"
            or not lock.get("lockable")
        ):
            raise SystemExit(f"Invalid Stage 7 task lock: {path}")
        if lock["model"] != model:
            raise SystemExit(f"Task lock model does not match {model}: {path}")
        if lock["preregistration_sha256"] != PREREGISTRATION_SHA256:
            raise SystemExit("Task lock preregistration hash does not match")
        task_name = str(lock["task"])
        if task_name in locks:
            raise SystemExit(f"Duplicate task lock for {task_name}")
        locks[task_name] = (path, lock)
    if set(locks) != set(TASK_NAMES):
        raise SystemExit("Admission requires both Stage 7 task locks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "generation_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    task_examples = {}
    task_contexts = {}
    prompts_by_task = {}
    for task_name, (_path, lock) in locks.items():
        task = make_task(task_name)
        examples = locked_examples(task_name, lock["selected_test_item_ids"])
        if examples_sha256(examples) != lock["selected_examples_sha256"]:
            raise SystemExit(f"Selected example hash mismatch for {task_name}")
        if task_dataset_sha256(task, examples) != lock["selected_dataset_sha256"]:
            raise SystemExit(f"Selected dataset hash mismatch for {task_name}")
        context = {}
        task_examples[task_name] = examples
        task_contexts[task_name] = context
        prompts_by_task[task_name] = [
            build_prompt(task, example, context, condition=CONDITION)
            for example in examples
        ]

    open_generator = None
    bedrock_generator = None
    if args.engine == "transformers":
        open_generator = SteeredGenerator(
            TransformersConfig(
                model=model,
                max_new_tokens=8,
                batch_size=args.batch_size,
                dtype=args.dtype,
                attention_implementation=args.attention_implementation,
                device_map=args.device_map,
            )
        )
    else:
        bedrock_generator = BedrockGenerator(
            model_id=model,
            region=str(frontier_lock["region"]),
            max_tokens=8,
            workers=args.workers,
        )

    root_metadata = {
        "version": 1,
        "stage": 7,
        "artifact": "held_out_admission_bundle",
        "model": model,
        "engine": args.engine,
        "tasks": {},
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }
    for task_name in TASK_NAMES:
        task = make_task(task_name)
        path, lock = locks[task_name]
        examples = task_examples[task_name]
        context = task_contexts[task_name]
        prompts = prompts_by_task[task_name]
        requires_held_out = task_requires_held_out_admission(lock)
        item_ids = [int(example.item_id) for example in examples]
        task_dir = args.output_dir / task_name
        task_dir.mkdir(exist_ok=True)
        prompt_rows = (
            [
                {
                    "task": task_name,
                    "condition": f"{CONDITION}__none",
                    "item_id": example.item_id,
                    **task.example_metadata(example),
                    "prompt": prompt,
                }
                for example, prompt in zip(examples, prompts, strict=True)
            ]
            if requires_held_out
            else []
        )
        write_jsonl(task_dir / "prompts.jsonl", prompt_rows)
        checkpoint = checkpoint_dir / f"{task_name}.jsonl"
        if requires_held_out:
            checkpoint_rows = _checkpoint_rows(checkpoint, item_ids)
            if checkpoint_rows is None:
                started = time()
                if open_generator is not None:
                    outputs = open_generator.generate(
                        open_generator.render_chat(prompts)
                    )
                    checkpoint_rows = [
                        {"item_id": example.item_id, "raw_output": raw_output}
                        for example, raw_output in zip(
                            examples,
                            outputs,
                            strict=True,
                        )
                    ]
                else:
                    results = bedrock_generator.generate(prompts)
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
                        for example, result in zip(
                            examples,
                            results,
                            strict=True,
                        )
                    ]
                elapsed = time() - started
                write_jsonl(checkpoint, checkpoint_rows)
            else:
                elapsed = 0.0
        else:
            checkpoint_rows = []
            elapsed = 0.0

        predictions = []
        prediction_examples = examples if requires_held_out else []
        for example, raw in zip(
            prediction_examples,
            checkpoint_rows,
            strict=True,
        ):
            raw_output = str(raw["raw_output"])
            parsed = task.parse(raw_output, example, context)
            row = {
                "task": task_name,
                "condition": f"{CONDITION}__none",
                "item_id": example.item_id,
                **task.example_metadata(example),
                "raw_output": raw_output,
                "pred": parsed.pred,
                "parse_method": parsed.parse_method,
                "details": parsed.details,
                **task.score(parsed, example),
            }
            if open_generator is not None:
                row["output_tokens"] = len(
                    open_generator.tokenizer.encode(
                        raw_output,
                        add_special_tokens=False,
                    )
                )
            else:
                row.update(
                    {
                        key: raw[key]
                        for key in (
                            "stop_reason",
                            "input_tokens",
                            "output_tokens",
                            "total_tokens",
                            "latency_ms",
                        )
                    }
                )
            predictions.append(row)
        write_jsonl(task_dir / "predictions.jsonl", predictions)

        rates = []
        admitted = []
        not_testable = []
        for signal, signal_lock in lock["signals"].items():
            dev_status = signal_lock["dev_status"]
            if dev_status == "calibrated":
                eligible = [
                    row
                    for row in predictions
                    if eligible_for_signal(task_name, signal, row)
                ]
                rate = sum(float(row[signal]) for row in eligible) / len(eligible)
                passes_test = (
                    float(signal_lock["minimum"])
                    <= rate
                    <= float(signal_lock["maximum"])
                )
                status = "admitted" if passes_test else "not_testable_test"
            else:
                eligible = []
                rate = None
                passes_test = None
                status = dev_status
            rates.append(
                {
                    "task": task_name,
                    "signal": signal,
                    "rate": rate,
                    "eligible_n": len(eligible),
                    "minimum": signal_lock["minimum"],
                    "maximum": signal_lock["maximum"],
                    "predicted_dev_rate": signal_lock["predicted_dev_rate"],
                    "dev_status": dev_status,
                    "passes_test": passes_test,
                    "status": status,
                }
            )
            (admitted if status == "admitted" else not_testable).append(signal)
        write_csv(task_dir / "admission_rates.csv", rates)
        admission = {
            "version": 1,
            "stage": 7,
            "artifact": "held_out_dv_admission",
            "task": task_name,
            "model": model,
            "engine": args.engine,
            "split": "test",
            "condition": CONDITION,
            "seed": args.seed,
            "held_out_evaluated": requires_held_out,
            "num_examples": len(predictions),
            "num_locked_examples": len(examples),
            "dataset_sha256": lock["selected_dataset_sha256"],
            "rates": rates,
            "admitted_signals": admitted,
            "not_testable_signals": not_testable,
            "difficulty_lock": lock,
            "difficulty_lock_path": str(path.resolve()),
            "difficulty_lock_sha256": sha256_file(path),
            "lockable": True,
            "elapsed_seconds": elapsed,
            "preregistration_sha256": PREREGISTRATION_SHA256,
        }
        for name, payload in (
            ("admission.json", admission),
            ("metadata.json", admission),
            (
                "validation.json",
                {
                    "status": "complete",
                    "stage": 7,
                    "task": task_name,
                    "model": model,
                    "held_out_evaluated": requires_held_out,
                    "num_prompts": len(prompt_rows),
                    "num_predictions": len(predictions),
                    "admitted_signals": admitted,
                    "not_testable_signals": not_testable,
                },
            ),
        ):
            (task_dir / name).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        root_metadata["tasks"][task_name] = {
            "admission_dir": str(task_dir.resolve()),
            "admitted_signals": admitted,
            "not_testable_signals": not_testable,
        }

    root_metadata["frontier_lock"] = frontier_lock
    root_metadata["frontier_lock_path"] = (
        str(args.frontier_lock.resolve()) if args.frontier_lock else None
    )
    root_metadata["frontier_lock_sha256"] = (
        sha256_file(args.frontier_lock) if args.frontier_lock else None
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(root_metadata, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "validation.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": 7,
                "artifact": "held_out_admission_bundle",
                "model": model,
                "tasks": list(TASK_NAMES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_environment_snapshots(args.output_dir)
    print(json.dumps(root_metadata, indent=2))


if __name__ == "__main__":
    main()
