"""Freeze one model's development-selected Stage 7 difficulty mix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from emotion_stage3.io import read_jsonl, sha256_file

from .constants import (
    ADMISSION_SPECS,
    MIXTURE_UNITS,
    PREREGISTRATION_SHA256,
    SEED,
    TARGET_BASE_RATE,
    TASK_NAMES,
)
from .tasks import (
    eligible_for_signal,
    examples_sha256,
    locked_examples,
    make_task,
    task_dataset_sha256,
)


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def select_tier_mix(
    tier_rates: list[float],
    *,
    target: float = TARGET_BASE_RATE,
    units: int = MIXTURE_UNITS,
) -> dict:
    best = None
    for counts in _compositions(units, len(tier_rates)):
        weights = [count / units for count in counts]
        predicted = sum(
            weight * rate
            for weight, rate in zip(weights, tier_rates, strict=True)
        )
        entropy = -sum(
            weight * math.log(weight)
            for weight in weights
            if weight > 0
        )
        key = (abs(predicted - target), -entropy, counts)
        if best is None or key < best[0]:
            best = (key, counts, predicted, entropy)
    if best is None:
        raise ValueError("No Stage 7 tier mixture candidates")
    _, counts, predicted, entropy = best
    return {
        "units": list(counts),
        "weights": [count / units for count in counts],
        "predicted_rate": predicted,
        "absolute_target_error": abs(predicted - target),
        "entropy": entropy,
    }


def _calibration_rows(calibration_dir: Path, task_name: str) -> list[dict]:
    rows = [
        row
        for row in read_jsonl(calibration_dir / "predictions.jsonl")
        if row["task"] == task_name
    ]
    if not rows:
        raise ValueError(f"Calibration has no rows for {task_name}")
    return rows


def _calibration_digest(calibration_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("metadata.json", "predictions.jsonl"):
        digest.update(bytes.fromhex(sha256_file(calibration_dir / name)))
    return digest.hexdigest()


def make_task_lock(calibration_dir: Path, task_name: str) -> dict:
    if task_name not in TASK_NAMES:
        raise ValueError(f"Unknown Stage 7 task {task_name!r}")
    metadata = json.loads(
        (calibration_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("stage") != 7 or metadata.get("artifact") != "difficulty_calibration":
        raise ValueError("Invalid Stage 7 calibration artifact")
    if metadata.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("Calibration preregistration hash does not match")
    task = make_task(task_name)
    dev_examples, _ = task.load("dev", None, SEED)
    test_examples, _ = task.load("test", None, SEED)
    if metadata["tasks"][task_name]["dev_dataset_sha256"] != task_dataset_sha256(
        task, dev_examples
    ):
        raise ValueError("Calibration development dataset hash does not match")
    rows = _calibration_rows(calibration_dir, task_name)
    row_ids = {int(row["item_id"]) for row in rows}
    if row_ids != {int(example.item_id) for example in dev_examples}:
        raise ValueError("Calibration rows do not cover the complete dev pool")

    tier_names = list(task.metadata()["difficulty_tiers"])
    signal_locks = {}
    selected_ids = []
    for signal, (minimum, maximum) in ADMISSION_SPECS[task_name].items():
        eligible = [
            row for row in rows if eligible_for_signal(task_name, signal, row)
        ]
        tier_rates = []
        tier_ns = []
        for tier in range(len(tier_names)):
            tier_rows = [
                row
                for row in eligible
                if int(row["difficulty_tier"]) == tier
            ]
            if not tier_rows:
                raise ValueError(f"No calibration rows for {task_name}/{signal}/tier {tier}")
            tier_ns.append(len(tier_rows))
            tier_rates.append(
                sum(float(row[signal]) for row in tier_rows) / len(tier_rows)
            )
        mixture = select_tier_mix(tier_rates)
        predicted = float(mixture["predicted_rate"])
        dev_eligible = minimum <= predicted <= maximum

        candidates = [
            example
            for example in test_examples
            if (
                task_name != "refusal_drift"
                or (
                    signal == "unsafe_compliance"
                    and bool(example.payload["should_refuse"])
                )
                or (
                    signal == "overrefusal"
                    and not bool(example.payload["should_refuse"])
                )
            )
        ]
        by_tier: dict[int, list] = {}
        for example in candidates:
            by_tier.setdefault(
                int(example.payload["difficulty_tier"]), []
            ).append(example)
        per_signal_target = 600 if task_name == "sycophancy" else 480
        ids = []
        tier_counts = []
        for tier, units in enumerate(mixture["units"]):
            count = int(units) * (per_signal_target // MIXTURE_UNITS)
            available = by_tier[tier]
            if len(available) < count:
                raise ValueError(
                    f"Tier {tier} has {len(available)} test items, needs {count}"
                )
            ids.extend(int(example.item_id) for example in available[:count])
            tier_counts.append(count)
        if len(ids) != per_signal_target:
            raise ValueError("Selected Stage 7 test count does not match target")
        selected_ids.extend(ids)
        signal_locks[signal] = {
            "signal": signal,
            "target_rate": TARGET_BASE_RATE,
            "minimum": minimum,
            "maximum": maximum,
            "tier_names": tier_names,
            "tier_dev_n": tier_ns,
            "tier_dev_rates": tier_rates,
            "mixture_units": mixture["units"],
            "mixture_weights": mixture["weights"],
            "selected_test_counts": tier_counts,
            "selected_test_item_ids": ids,
            "predicted_dev_rate": predicted,
            "absolute_target_error": mixture["absolute_target_error"],
            "mixture_entropy": mixture["entropy"],
            "dev_status": "calibrated" if dev_eligible else "not_testable_dev",
        }

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Signals selected overlapping Stage 7 test item IDs")
    selected = locked_examples(task_name, selected_ids)
    return {
        "version": 1,
        "stage": 7,
        "artifact": "difficulty_lock",
        "model": metadata["model"],
        "engine": metadata["engine"],
        "task": task_name,
        "seed": SEED,
        "calibration_condition": "neutral_length_matched",
        "target_rate": TARGET_BASE_RATE,
        "mixture_increment": 1 / MIXTURE_UNITS,
        "selection_rule": (
            "development-only 0.05-grid mixture nearest 0.30; ties by higher "
            "entropy then lexicographic tier units"
        ),
        "signals": signal_locks,
        "selected_test_item_ids": selected_ids,
        "num_selected_test_items": len(selected_ids),
        "selected_examples_sha256": examples_sha256(selected),
        "selected_dataset_sha256": task_dataset_sha256(task, selected),
        "full_dev_dataset_sha256": task_dataset_sha256(task, dev_examples),
        "full_test_pool_sha256": task_dataset_sha256(task, test_examples),
        "task_configuration": task.metadata(),
        "calibration_dir": str(calibration_dir.resolve()),
        "calibration_sha256": _calibration_digest(calibration_dir),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "lockable": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    lock = make_task_lock(args.calibration_dir, args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
