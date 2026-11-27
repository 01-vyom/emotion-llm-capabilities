"""Run one model's Stage 7 development difficulty calibration."""

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

from .constants import (
    ADMISSION_SPECS,
    PREREGISTRATION_SHA256,
    SEED,
    TASK_NAMES,
)
from .tasks import (
    eligible_for_signal,
    make_task,
    task_dataset_sha256,
)


CONDITION = "neutral_length_matched"


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
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
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
            raise SystemExit("Bedrock calibration requires --frontier-lock")
        frontier_lock = json.loads(args.frontier_lock.read_text(encoding="utf-8"))
        model = str(frontier_lock["model_id"])
    else:
        if not args.model:
            raise SystemExit("Transformers calibration requires --model")
        model = args.model

    task_records = {}
    task_examples = {}
    task_contexts = {}
    prompts_by_task = {}
    prompt_rows = []
    for task_name in args.tasks:
        task = make_task(task_name)
        examples, context = task.load("dev", None, args.seed)
        prompts = [
            build_prompt(task, example, context, condition=CONDITION)
            for example in examples
        ]
        task_examples[task_name] = examples
        task_contexts[task_name] = context
        prompts_by_task[task_name] = prompts
        task_records[task_name] = {
            "num_dev_examples": len(examples),
            "dev_dataset_sha256": task_dataset_sha256(task, examples),
            "task_configuration": task.metadata(),
        }
        prompt_rows.extend(
            {
                "task": task_name,
                "condition": f"{CONDITION}__none",
                "item_id": example.item_id,
                **task.example_metadata(example),
                "prompt": prompt,
            }
            for example, prompt in zip(examples, prompts, strict=True)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "generation_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    write_jsonl(args.output_dir / "prompts.jsonl", prompt_rows)

    open_generator = None
    bedrock_generator = None
    max_rendered_prompt_tokens = None
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
        rendered = [
            prompt
            for task_name in args.tasks
            for prompt in open_generator.render_chat(prompts_by_task[task_name])
        ]
        max_rendered_prompt_tokens = max(
            len(
                open_generator.tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                )
            )
            for prompt in rendered
        )
    else:
        bedrock_generator = BedrockGenerator(
            model_id=model,
            region=str(frontier_lock["region"]),
            max_tokens=8,
            workers=args.workers,
        )

    predictions = []
    elapsed_by_task = {}
    for task_name in args.tasks:
        task = make_task(task_name)
        examples = task_examples[task_name]
        context = task_contexts[task_name]
        prompts = prompts_by_task[task_name]
        item_ids = [int(example.item_id) for example in examples]
        checkpoint = checkpoint_dir / f"{task_name}.jsonl"
        checkpoint_rows = _checkpoint_rows(checkpoint, item_ids)
        if checkpoint_rows is None:
            started = time()
            if open_generator is not None:
                outputs = open_generator.generate(
                    open_generator.render_chat(prompts)
                )
                checkpoint_rows = [
                    {
                        "item_id": example.item_id,
                        "raw_output": raw_output,
                    }
                    for example, raw_output in zip(examples, outputs, strict=True)
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
                    for example, result in zip(examples, results, strict=True)
                ]
            elapsed_by_task[task_name] = time() - started
            write_jsonl(checkpoint, checkpoint_rows)
        else:
            elapsed_by_task[task_name] = 0.0

        for example, raw in zip(examples, checkpoint_rows, strict=True):
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

    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    rate_rows = []
    for task_name in args.tasks:
        for signal in ADMISSION_SPECS[task_name]:
            eligible = [
                row
                for row in predictions
                if row["task"] == task_name
                and eligible_for_signal(task_name, signal, row)
            ]
            for tier in sorted({int(row["difficulty_tier"]) for row in eligible}):
                tier_rows = [
                    row
                    for row in eligible
                    if int(row["difficulty_tier"]) == tier
                ]
                rate_rows.append(
                    {
                        "task": task_name,
                        "signal": signal,
                        "difficulty_tier": tier,
                        "n": len(tier_rows),
                        "rate": sum(float(row[signal]) for row in tier_rows)
                        / len(tier_rows),
                    }
                )
    write_csv(args.output_dir / "tier_rates.csv", rate_rows)
    metadata = {
        "version": 1,
        "stage": 7,
        "artifact": "difficulty_calibration",
        "model": model,
        "engine": args.engine,
        "split": "dev",
        "condition": CONDITION,
        "seed": args.seed,
        "tasks": task_records,
        "num_prompts": len(prompt_rows),
        "batch_size": args.batch_size if open_generator else None,
        "workers": args.workers if bedrock_generator else None,
        "dtype": args.dtype if open_generator else None,
        "attention_implementation": (
            args.attention_implementation if open_generator else None
        ),
        "device_map": args.device_map if open_generator else None,
        "max_new_tokens": 8,
        "max_rendered_prompt_tokens": max_rendered_prompt_tokens,
        "frontier_lock": frontier_lock,
        "frontier_lock_path": (
            str(args.frontier_lock.resolve()) if args.frontier_lock else None
        ),
        "frontier_lock_sha256": (
            sha256_file(args.frontier_lock) if args.frontier_lock else None
        ),
        "elapsed_seconds_by_task": elapsed_by_task,
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    validation = {
        "status": "complete",
        "stage": 7,
        "artifact": "difficulty_calibration",
        "model": model,
        "engine": args.engine,
        "num_prompts": len(prompt_rows),
        "num_predictions": len(predictions),
        "tasks": list(args.tasks),
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )
    write_environment_snapshots(args.output_dir)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

