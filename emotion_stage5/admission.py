"""Run the preregistered Stage 5 no-steer DV admission gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

from emotion_stage2.tasks.base import build_prompt
from emotion_stage3.io import write_csv, write_environment_snapshots, write_jsonl
from emotion_stage3.steering import SteeredGenerator, TransformersConfig

from .tasks import (
    ADMISSION_SPECS,
    TASK_NAMES,
    eligible_for_signal,
    make_task,
    task_dataset_sha256,
)


ADMISSION_CONDITION = "neutral_length_matched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--device-map")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {args.output_dir}")
    task = make_task(args.task)
    examples, task_context = task.load(args.split, args.limit, args.seed)
    prompts = [
        build_prompt(
            task,
            example,
            task_context,
            condition=ADMISSION_CONDITION,
        )
        for example in examples
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "prompts.jsonl",
        [
            {
                "condition": ADMISSION_CONDITION,
                "item_id": example.item_id,
                **task.example_metadata(example),
                "prompt": prompt,
            }
            for example, prompt in zip(examples, prompts, strict=True)
        ],
    )

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
    rendered = generator.render_chat(prompts)
    started = time()
    outputs = generator.generate(rendered)
    elapsed = time() - started

    predictions = []
    for example, raw_output in zip(examples, outputs, strict=True):
        parsed = task.parse(raw_output, example, task_context)
        scores = task.score(parsed, example)
        predictions.append(
            {
                "condition": ADMISSION_CONDITION,
                "item_id": example.item_id,
                **task.example_metadata(example),
                "raw_output": raw_output,
                "pred": parsed.pred,
                "parse_method": parsed.parse_method,
                "details": parsed.details,
                **scores,
            }
        )
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)

    rate_rows = []
    admitted = []
    not_testable = []
    for signal, (minimum, maximum) in ADMISSION_SPECS[task.name].items():
        eligible = [
            row
            for row in predictions
            if eligible_for_signal(task.name, signal, row)
        ]
        rate = (
            sum(float(row[signal]) for row in eligible) / len(eligible)
            if eligible
            else 0.0
        )
        passes = minimum <= rate <= maximum
        rate_rows.append(
            {
                "task": task.name,
                "signal": signal,
                "rate": rate,
                "eligible_n": len(eligible),
                "minimum": minimum,
                "maximum": maximum,
                "passes": passes,
            }
        )
        (admitted if passes else not_testable).append(signal)
    write_csv(args.output_dir / "admission_rates.csv", rate_rows)

    dataset_hash = task_dataset_sha256(task, examples)
    admission = {
        "version": 1,
        "stage": 5,
        "artifact": "dv_admission",
        "task": task.name,
        "model": args.model,
        "split": args.split,
        "condition": ADMISSION_CONDITION,
        "seed": args.seed,
        "num_examples": len(examples),
        "dataset_sha256": dataset_hash,
        "rates": rate_rows,
        "admitted_signals": admitted,
        "not_testable_signals": not_testable,
        "lockable": args.split == "test" and args.limit is None,
        "task_configuration": task.metadata(),
        "elapsed_seconds": elapsed,
    }
    (args.output_dir / "admission.json").write_text(
        json.dumps(admission, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                **admission,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "attention_implementation": args.attention_implementation,
                "device_map": args.device_map,
                "max_new_tokens": task.max_new_tokens(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "validation.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "task": task.name,
                "model": args.model,
                "split": args.split,
                "num_prompts": len(prompts),
                "num_predictions": len(predictions),
                "admitted_signals": admitted,
                "not_testable_signals": not_testable,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_environment_snapshots(args.output_dir)
    print(json.dumps(admission, indent=2))


if __name__ == "__main__":
    main()
