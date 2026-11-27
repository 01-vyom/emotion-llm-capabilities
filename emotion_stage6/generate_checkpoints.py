"""Generate a disjoint Stage 6 arm range into an existing checkpoint directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from emotion_stage2.tasks.base import build_prompt
from emotion_stage3.io import read_jsonl, write_jsonl
from emotion_stage3.run_stage3 import load_steering_spec
from emotion_stage3.steering import SteeredGenerator, TransformersConfig

from .run_stage6 import _validate_locks, build_stage6_arms
from .tasks import TASK_NAMES, make_task, task_dataset_sha256


def _complete_checkpoint(path: Path, item_ids: list[int]) -> bool:
    if not path.exists():
        return False
    rows = read_jsonl(path)
    return (
        [int(row["item_id"]) for row in rows] == item_ids
        and all("raw_output" in row for row in rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--panel-lock", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--steering-lock", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--stop-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    task = make_task(args.task)
    examples, task_context = task.load("test", None, 13)
    panel_lock, steering_lock, _admitted = _validate_locks(
        panel_lock_path=args.panel_lock,
        steering_lock_path=args.steering_lock,
        model=args.model,
        task_name=task.name,
        vector_dir=args.vector_dir,
        dataset_hash=task_dataset_sha256(task, examples),
        prompt_only=False,
    )
    del panel_lock
    if steering_lock is None:
        raise SystemExit("Checkpoint workers require a steering lock")
    arms = build_stage6_arms(task.name)
    if not 0 <= args.start_index < args.stop_index <= len(arms):
        raise SystemExit(
            f"Arm range must be within [0, {len(arms)}]: "
            f"{args.start_index}:{args.stop_index}"
        )
    selected = arms[args.start_index : args.stop_index]
    checkpoint_dir = args.output_dir / "generation_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    item_ids = [example.item_id for example in examples]
    pending = [
        arm
        for arm in selected
        if not _complete_checkpoint(
            checkpoint_dir / f"{arm.name}.jsonl",
            item_ids,
        )
    ]
    if not pending:
        return

    generator = SteeredGenerator(
        TransformersConfig(
            model=args.model,
            max_new_tokens=task.max_new_tokens(),
            batch_size=args.batch_size,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
        )
    )
    direction = steering_lock["direction"]
    policy = steering_lock["policy"]
    layers = tuple(int(value) for value in steering_lock["layers"])
    for arm in pending:
        checkpoint = checkpoint_dir / f"{arm.name}.jsonl"
        if _complete_checkpoint(checkpoint, item_ids):
            continue
        prompts = [
            build_prompt(
                task,
                example,
                task_context,
                condition=arm.prompt_condition,
            )
            for example in examples
        ]
        spec = load_steering_spec(
            args.vector_dir,
            pair_name=direction,
            policy=policy,
            layers=layers,
            vector_kind=arm.vector_kind,
            alpha=arm.alpha,
        )
        outputs = generator.generate(
            generator.render_chat(prompts),
            steering=spec,
        )
        rows = [
            {"item_id": example.item_id, "raw_output": raw_output}
            for example, raw_output in zip(examples, outputs, strict=True)
        ]
        temporary = checkpoint.with_name(
            f".{checkpoint.name}.tmp.{os.getpid()}"
        )
        write_jsonl(temporary, rows)
        temporary.replace(checkpoint)
        print(f"completed {arm.name}", flush=True)


if __name__ == "__main__":
    main()
