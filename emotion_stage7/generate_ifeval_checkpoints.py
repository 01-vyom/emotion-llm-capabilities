"""Generate a disjoint Stage 7 IFEval arm range into an existing run."""

from __future__ import annotations

import argparse
from pathlib import Path

from emotion_stage2.tasks.base import build_prompt
from emotion_stage2.tasks.ifeval import IFEvalTask
from emotion_stage3.run_stage3 import load_steering_spec
from emotion_stage3.steering import SteeredGenerator, TransformersConfig

from .arms import ifeval_arms
from .constants import IFEVAL_MAX_NEW_TOKENS, SEED
from .io import checkpoint_rows, load_steering_lock, write_jsonl_atomic


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--steering-lock", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--stop-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--device-map")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    arms = ifeval_arms()
    if not 0 <= args.start_index < args.stop_index <= len(arms):
        raise SystemExit(
            f"Arm range must be within [0, {len(arms)}]: "
            f"{args.start_index}:{args.stop_index}"
        )

    lock = load_steering_lock(
        args.steering_lock,
        vector_dir=args.vector_dir,
        model=args.model,
    )
    task = IFEvalTask()
    examples, context = task.load("test", None, args.seed)
    item_ids = [int(example.item_id) for example in examples]
    checkpoint_dir = args.output_dir / "generation_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selected = arms[args.start_index : args.stop_index]
    pending = [
        arm
        for arm in selected
        if checkpoint_rows(
            checkpoint_dir / f"{arm.name}.jsonl",
            item_ids,
        )
        is None
    ]
    if not pending:
        return

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
    layers = tuple(int(value) for value in lock["layers"])
    for arm in pending:
        checkpoint = checkpoint_dir / f"{arm.name}.jsonl"
        if checkpoint_rows(checkpoint, item_ids) is not None:
            continue
        prompts = [
            build_prompt(
                task,
                example,
                context,
                condition=arm.prompt_condition,
            )
            for example in examples
        ]
        spec = (
            load_steering_spec(
                args.vector_dir,
                pair_name=lock["direction"],
                policy=lock["policy"],
                layers=layers,
                vector_kind=arm.vector_kind,
                alpha=arm.alpha,
            )
            if arm.vector_kind is not None
            else None
        )
        outputs = generator.generate(
            generator.render_chat(prompts),
            steering=spec,
        )
        write_jsonl_atomic(
            checkpoint,
            (
                {
                    "item_id": example.item_id,
                    "raw_output": raw_output,
                }
                for example, raw_output in zip(
                    examples,
                    outputs,
                    strict=True,
                )
            ),
        )
        print(f"completed {arm.name}", flush=True)


if __name__ == "__main__":
    main()
