"""Run a Stage 6 frontier no-steer DV admission gate through Bedrock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

from emotion_stage2.tasks.base import build_prompt
from emotion_stage3.io import (
    sha256_file,
    write_csv,
    write_environment_snapshots,
    write_jsonl,
)

from .admission import ADMISSION_CONDITION
from .frontier import BedrockGenerator
from .tasks import (
    ADMISSION_SPECS,
    TASK_NAMES,
    eligible_for_signal,
    make_task,
    task_dataset_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--frontier-lock", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {args.output_dir}")
    lock = json.loads(args.frontier_lock.read_text(encoding="utf-8"))
    if lock.get("stage") != 6 or lock.get("artifact") != "frontier_model_lock":
        raise SystemExit("Invalid frontier lock")
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
    generator = BedrockGenerator(
        model_id=lock["model_id"],
        region=lock["region"],
        max_tokens=args.max_tokens,
        workers=args.workers,
    )
    started = time()
    results = generator.generate(prompts)
    elapsed = time() - started
    predictions = []
    for example, result in zip(examples, results, strict=True):
        parsed = task.parse(result.raw_output, example, task_context)
        predictions.append(
            {
                "condition": ADMISSION_CONDITION,
                "item_id": example.item_id,
                **task.example_metadata(example),
                "raw_output": result.raw_output,
                "pred": parsed.pred,
                "parse_method": parsed.parse_method,
                "details": parsed.details,
                **task.score(parsed, example),
                "stop_reason": result.stop_reason,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "latency_ms": result.latency_ms,
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
    admission = {
        "version": 1,
        "stage": 6,
        "artifact": "dv_admission",
        "task": task.name,
        "model": lock["model_id"],
        "engine": "bedrock",
        "split": args.split,
        "condition": ADMISSION_CONDITION,
        "seed": args.seed,
        "num_examples": len(examples),
        "dataset_sha256": task_dataset_sha256(task, examples),
        "rates": rate_rows,
        "admitted_signals": admitted,
        "not_testable_signals": not_testable,
        "lockable": args.split == "test" and args.limit is None,
        "task_configuration": task.metadata(),
        "elapsed_seconds": elapsed,
        "frontier_lock_path": str(args.frontier_lock.resolve()),
        "frontier_lock_sha256": sha256_file(args.frontier_lock),
    }
    (args.output_dir / "admission.json").write_text(
        json.dumps(admission, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                **admission,
                "workers": args.workers,
                "max_new_tokens": args.max_tokens,
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
                "model": lock["model_id"],
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
