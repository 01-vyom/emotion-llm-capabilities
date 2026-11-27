"""Run Stage 3 baseline, calibration, or locked steering experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from emotion_stage2.metrics import paired_delta_dicts, summarize_signals
from emotion_stage2.tasks.base import build_prompt
from emotion_stage2.tasks.registry import TASK_NAMES, make_task

from .io import (
    read_jsonl,
    sha256_file,
    write_csv,
    write_environment_snapshots,
    write_jsonl,
)
from .steering import (
    SteeredGenerator,
    SteeringSpec,
    TransformersConfig,
)
from .vectors import POSITION_POLICIES


COMMON_SIGNAL_KEYS = ("output_tokens", "output_truncated")


@dataclass(frozen=True)
class Arm:
    name: str
    prompt_condition: str
    vector_kind: str | None
    alpha: float


def _alpha_label(alpha: float) -> str:
    return f"{alpha:+.4f}"


def build_arms(
    design: str,
    *,
    alpha: float | None = None,
    calibration_alphas: tuple[float, ...] = (),
    affect_condition: str = "calm_meticulous",
) -> tuple[list[Arm], list[str]]:
    if design == "baseline":
        return [Arm("no_steer", "neutral_empty", None, 0.0)], ["no_steer"]
    if design == "calibration":
        magnitudes = sorted({abs(value) for value in calibration_alphas if value})
        if not magnitudes:
            raise ValueError("Calibration requires at least one non-zero alpha")
        arms = [Arm("no_steer", "neutral_empty", None, 0.0)]
        for magnitude in magnitudes:
            label = _alpha_label(magnitude)
            arms.extend(
                [
                    Arm(f"positive_a{label}", "neutral_empty", "unit", magnitude),
                    Arm(f"negative_a-{label[1:]}", "neutral_empty", "unit", -magnitude),
                    Arm(f"random_a{label}", "neutral_empty", "random_unit", magnitude),
                    Arm(
                        f"shuffled_a{label}",
                        "neutral_empty",
                        "shuffled_unit",
                        magnitude,
                    ),
                ]
            )
        return arms, ["no_steer"]
    if design == "confirmatory":
        if alpha is None or alpha <= 0:
            raise ValueError("Confirmatory design requires a positive locked alpha")
        label = _alpha_label(alpha)
        random_name = f"random_a{label}"
        return [
            Arm("no_steer", "neutral_empty", None, 0.0),
            Arm(f"positive_a{label}", "neutral_empty", "unit", alpha),
            Arm(f"negative_a-{label[1:]}", "neutral_empty", "unit", -alpha),
            Arm(random_name, "neutral_empty", "random_unit", alpha),
            Arm(
                f"shuffled_a{label}",
                "neutral_empty",
                "shuffled_unit",
                alpha,
            ),
            Arm(
                f"affect_opposite_a-{label[1:]}",
                affect_condition,
                "unit",
                -alpha,
            ),
            Arm(
                f"affect_matching_a{label}",
                affect_condition,
                "unit",
                alpha,
            ),
        ], ["no_steer", random_name]
    raise ValueError(f"Unknown design {design!r}")


def load_steering_spec(
    vector_dir: Path,
    *,
    pair_name: str,
    policy: str,
    layers: tuple[int, ...],
    vector_kind: str,
    alpha: float,
) -> SteeringSpec:
    vectors = np.load(vector_dir / "vectors.npz")
    available_pairs = [str(value) for value in vectors["pair_names"]]
    available_layers = [int(value) for value in vectors["layers"]]
    if pair_name not in available_pairs:
        raise ValueError(
            f"Unknown direction {pair_name!r}; available: {available_pairs}"
        )
    if policy not in POSITION_POLICIES:
        raise ValueError(f"Unknown position policy {policy!r}")
    key = f"{policy}_{vector_kind}"
    if key not in vectors:
        raise ValueError(f"Vector artifact does not contain {key!r}")
    pair_index = available_pairs.index(pair_name)
    layer_offsets = {}
    for layer in layers:
        if layer not in available_layers:
            raise ValueError(
                f"Layer {layer} is not in vector artifact {available_layers}"
            )
        layer_offsets[layer] = available_layers.index(layer)
    unit_array = vectors[key][pair_index]
    residual_norms = vectors["residual_norm"]
    return SteeringSpec(
        vectors={
            layer: unit_array[offset]
            for layer, offset in layer_offsets.items()
        },
        residual_norms={
            layer: float(residual_norms[offset])
            for layer, offset in layer_offsets.items()
        },
        alpha=alpha,
        position_policy="last",
    )


def _load_lock(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_lock(
    lock: dict[str, Any],
    *,
    model: str,
    vector_dir: Path,
) -> None:
    if lock.get("version") not in {1, 2}:
        raise SystemExit("Steering lock must have version 1 or 2")
    if lock.get("model") != model:
        raise SystemExit("Steering lock model does not match --model")
    expected_sha = sha256_file(vector_dir / "vectors.npz")
    if lock.get("vectors_sha256") != expected_sha:
        raise SystemExit("Steering lock vectors_sha256 does not match the artifact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--design",
        choices=("baseline", "calibration", "confirmatory"),
        required=True,
    )
    parser.add_argument("--stage", choices=("3", "3.5"), default="3")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--vector-dir", type=Path)
    parser.add_argument("--direction", default="diligence")
    parser.add_argument(
        "--policy",
        choices=POSITION_POLICIES,
        default="mean",
    )
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--calibration-alphas", nargs="+", type=float, default=[])
    parser.add_argument("--steering-lock", type=Path)
    parser.add_argument("--affect-condition", default="calm_meticulous")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--device-map")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--enable-yarn", action="store_true")
    parser.add_argument(
        "--yarn-original-max-position-embeddings",
        type=int,
        default=40960,
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--fixture-smoke-test", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete per-arm generation checkpoints in --output-dir.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ruler-context-lengths",
        nargs="+",
        type=int,
        default=[32768, 65536],
    )
    parser.add_argument(
        "--ruler-depths",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    parser.add_argument(
        "--ruler-modes",
        nargs="+",
        default=["single", "multi", "aggregate"],
    )
    parser.add_argument("--ruler-samples-per-cell", type=int, default=1)
    parser.add_argument(
        "--ruler-hard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ruler-distractors", type=int, default=3)
    parser.add_argument("--ruler-multi-key-count", type=int, default=5)
    parser.add_argument("--ruler-aggregate-match-count", type=int, default=2)
    parser.add_argument("--ruler-context-reserve-tokens", type=int, default=1024)
    return parser.parse_args()


def _rope_scaling(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.enable_yarn:
        return None
    if args.max_model_len is None:
        raise SystemExit("--enable-yarn requires --max-model-len")
    factor = (
        args.max_model_len
        / args.yarn_original_max_position_embeddings
    )
    if factor <= 1:
        raise SystemExit("YaRN factor must be greater than 1")
    return {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": (
            args.yarn_original_max_position_embeddings
        ),
    }


def main() -> None:
    args = parse_args()
    output_has_files = (
        any(args.output_dir.iterdir()) if args.output_dir.exists() else False
    )
    if output_has_files and not args.resume:
        raise SystemExit(f"Refusing to overwrite non-empty {args.output_dir}")
    if args.design != "baseline" and args.vector_dir is None:
        raise SystemExit("Steered designs require --vector-dir")

    lock = None
    direction = args.direction
    policy = args.policy
    layers = tuple(args.layers or ())
    alpha = args.alpha
    if args.steering_lock is not None:
        if args.vector_dir is None:
            raise SystemExit("--steering-lock requires --vector-dir")
        lock = _load_lock(args.steering_lock)
        _validate_lock(lock, model=args.model, vector_dir=args.vector_dir)
        direction = lock["direction"]
        policy = lock["policy"]
        layers = tuple(int(value) for value in lock["layers"])
        alpha = float(lock["alpha"])
    if args.design == "confirmatory":
        if args.split == "test" and lock is None:
            raise SystemExit("Held-out confirmatory runs require --steering-lock")
    elif args.steering_lock is not None:
        raise SystemExit("--steering-lock is only valid for confirmatory runs")
    if args.design != "baseline" and not layers:
        raise SystemExit("Steered designs require --layers or a steering lock")

    arms, baseline_arms = build_arms(
        args.design,
        alpha=alpha,
        calibration_alphas=tuple(args.calibration_alphas),
        affect_condition=args.affect_condition,
    )
    task = make_task(args.task, args)
    if args.fixture_smoke_test:
        examples, task_context = task.load_fixture()
    else:
        examples, task_context = task.load(args.split, args.limit, args.seed)
    max_new_tokens = args.max_new_tokens or task.max_new_tokens()
    rope_scaling = _rope_scaling(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows = []
    raw_prompts_by_arm: dict[str, list[str]] = {}
    for arm in arms:
        arm_prompts = []
        for example in examples:
            prompt = build_prompt(
                task,
                example,
                task_context,
                condition=arm.prompt_condition,
            )
            arm_prompts.append(prompt)
            prompt_rows.append(
                {
                    "condition": arm.name,
                    "base_condition": arm.prompt_condition,
                    "item_id": example.item_id,
                    "vector_kind": arm.vector_kind,
                    "alpha": arm.alpha,
                    "direction": direction if arm.vector_kind else None,
                    "policy": policy if arm.vector_kind else None,
                    "layers": list(layers) if arm.vector_kind else [],
                    **task.example_metadata(example),
                    "prompt": prompt,
                }
            )
        raw_prompts_by_arm[arm.name] = arm_prompts
    write_jsonl(args.output_dir / "prompts.jsonl", prompt_rows)

    generator = SteeredGenerator(
        TransformersConfig(
            model=args.model,
            max_new_tokens=max_new_tokens,
            batch_size=args.batch_size,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            device_map=args.device_map,
            max_model_len=args.max_model_len,
            rope_scaling=rope_scaling,
        )
    )
    rendered_by_arm = {
        arm.name: generator.render_chat(raw_prompts_by_arm[arm.name])
        for arm in arms
    }
    prompt_lengths = [
        len(generator.tokenizer.encode(prompt, add_special_tokens=False))
        for prompts in rendered_by_arm.values()
        for prompt in prompts
    ]
    max_prompt_tokens = max(prompt_lengths, default=0)
    if (
        args.max_model_len is not None
        and max_prompt_tokens + max_new_tokens > args.max_model_len
    ):
        raise SystemExit(
            f"Rendered prompt {max_prompt_tokens} + generation "
            f"{max_new_tokens} exceeds max_model_len {args.max_model_len}"
        )

    prompt_row_by_arm_item = {
        (row["condition"], int(row["item_id"])): row
        for row in prompt_rows
    }
    prediction_rows = []
    arm_times = {}
    checkpoint_dir = args.output_dir / "generation_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time()
    for arm in arms:
        spec = None
        if arm.vector_kind is not None:
            spec = load_steering_spec(
                args.vector_dir,
                pair_name=direction,
                policy=policy,
                layers=layers,
                vector_kind=arm.vector_kind,
                alpha=arm.alpha,
            )
        checkpoint_path = checkpoint_dir / f"{arm.name}.jsonl"
        outputs = None
        if args.resume and checkpoint_path.exists():
            checkpoint_rows = read_jsonl(checkpoint_path)
            if (
                len(checkpoint_rows) == len(examples)
                and all(
                    row.get("condition") == arm.name
                    and int(row.get("item_id")) == example.item_id
                    for row, example in zip(
                        checkpoint_rows,
                        examples,
                        strict=True,
                    )
                )
            ):
                outputs = [str(row["raw_output"]) for row in checkpoint_rows]
                arm_times[arm.name] = 0.0
        if outputs is None:
            arm_started = time()
            outputs = generator.generate(rendered_by_arm[arm.name], steering=spec)
            arm_times[arm.name] = time() - arm_started
        if len(outputs) != len(examples):
            raise RuntimeError(
                f"Arm {arm.name} expected {len(examples)} outputs, "
                f"received {len(outputs)}"
            )
        write_jsonl(
            checkpoint_path,
            (
                {
                    "condition": arm.name,
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
        for example, raw_output in zip(examples, outputs, strict=True):
            prompt_row = prompt_row_by_arm_item[(arm.name, example.item_id)]
            parsed = task.parse(raw_output, example, task_context)
            token_count = len(
                generator.tokenizer.encode(
                    raw_output,
                    add_special_tokens=False,
                )
            )
            prediction_rows.append(
                {
                    key: value
                    for key, value in prompt_row.items()
                    if key != "prompt"
                }
                | {
                    "raw_output": raw_output,
                    "pred": parsed.pred,
                    "parse_method": parsed.parse_method,
                    "details": parsed.details,
                    **task.score(parsed, example),
                    "output_tokens": token_count,
                    "output_truncated": float(token_count >= max_new_tokens),
                }
            )
    elapsed = time() - started

    signal_keys = list(task.signal_keys) + list(COMMON_SIGNAL_KEYS)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_csv(
        args.output_dir / "summary.csv",
        summarize_signals(
            prediction_rows,
            signal_keys=signal_keys,
            seed=args.seed,
            reps=args.bootstrap_reps,
        ),
    )
    paired_rows = paired_delta_dicts(
        prediction_rows,
        signal_keys=signal_keys,
        baselines=baseline_arms,
        primary_signals=task.primary_signals,
        seed=args.seed,
        reps=args.bootstrap_reps,
    )
    for row in paired_rows:
        row["confirmatory_family"] = bool(
            args.design == "confirmatory"
            and row["signal"] in task.primary_signals
            and (
                row["condition"].startswith("positive_")
                or row["condition"].startswith("negative_")
            )
        )
        row["dose_family"] = bool(
            args.design == "calibration"
            and row["signal"] in task.primary_signals
            and (
                row["condition"].startswith("positive_")
                or row["condition"].startswith("negative_")
            )
        )
    write_csv(args.output_dir / "paired_deltas.csv", paired_rows)
    write_jsonl(
        args.output_dir / "confusions.jsonl",
        task.confusion_counts(prediction_rows),
    )

    vector_meta = None
    vectors_sha256 = None
    if args.vector_dir is not None:
        vector_meta = json.loads(
            (args.vector_dir / "extraction_meta.json").read_text(encoding="utf-8")
        )
        vectors_sha256 = sha256_file(args.vector_dir / "vectors.npz")
    metadata = {
        "stage": 3 if args.stage == "3" else 3.5,
        "task": task.name,
        "model": args.model,
        "engine": "transformers",
        "design": args.design,
        "split": args.split,
        "fixture_smoke_test": args.fixture_smoke_test,
        "num_examples": len(examples),
        "num_prompts": len(prompt_rows),
        "seed": args.seed,
        "arms": [arm.name for arm in arms],
        "paired_baselines": baseline_arms,
        "signal_keys": signal_keys,
        "primary_hypothesis_signals": list(task.primary_signals),
        "direction": direction,
        "policy": policy,
        "layers": list(layers),
        "alpha": alpha,
        "calibration_alphas": args.calibration_alphas,
        "affect_condition": args.affect_condition,
        "steering_lock": lock,
        "steering_lock_path": (
            str(args.steering_lock.resolve())
            if args.steering_lock is not None
            else None
        ),
        "vector_dir": (
            str(args.vector_dir.resolve())
            if args.vector_dir is not None
            else None
        ),
        "vectors_sha256": vectors_sha256,
        "vector_metadata": vector_meta,
        "alpha_parameterization": (
            "injected_delta = alpha * mean_training_token_residual_norm "
            "* unit_direction"
        ),
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "device_map": args.device_map,
        "batch_size": args.batch_size,
        "max_new_tokens": max_new_tokens,
        "max_model_len": args.max_model_len,
        "max_rendered_prompt_tokens": max_prompt_tokens,
        "rope_scaling": rope_scaling,
        "bootstrap_reps": args.bootstrap_reps,
        "task_context": task_context,
        "task_configuration": task.metadata(),
        **task.metadata(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "run_stats.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "num_generations": len(prediction_rows),
                "generations_per_second": (
                    len(prediction_rows) / elapsed if elapsed else None
                ),
                "total_output_tokens": sum(
                    int(row["output_tokens"]) for row in prediction_rows
                ),
                "arm_elapsed_seconds": arm_times,
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
                "num_predictions": len(prediction_rows),
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
