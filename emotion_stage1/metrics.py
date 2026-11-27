"""Metrics and confidence intervals for Stage 1 outputs."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConditionMetrics:
    condition: str
    n: int
    accuracy: float
    parse_rate: float
    macro_f1: float
    accuracy_ci_low: float
    accuracy_ci_high: float


@dataclass(frozen=True)
class PairedAccuracyDelta:
    condition: str
    baseline: str
    n: int
    accuracy_delta: float
    accuracy_delta_ci_low: float
    accuracy_delta_ci_high: float
    wins: int
    ties: int
    losses: int


def macro_f1(rows: list[dict], labels: Iterable[str] | None = None) -> float:
    evaluation_labels = (
        sorted(set(labels))
        if labels is not None
        else sorted(
            {row["gold_label"] for row in rows}
            | {row["pred_label"] for row in rows if row["pred_label"]}
        )
    )
    if not evaluation_labels:
        return 0.0

    f1s = []
    for label in evaluation_labels:
        tp = sum(row["gold_label"] == label and row["pred_label"] == label for row in rows)
        fp = sum(row["gold_label"] != label and row["pred_label"] == label for row in rows)
        fn = sum(row["gold_label"] == label and row["pred_label"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def bootstrap_accuracy_ci(rows: list[dict], seed: int = 13, reps: int = 2000) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    rng = random.Random(seed)
    values = []
    n = len(rows)
    for _ in range(reps):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        values.append(sum(row["correct"] for row in sample) / n)
    values.sort()
    lo = values[int(0.025 * (reps - 1))]
    hi = values[int(0.975 * (reps - 1))]
    return lo, hi


def summarize(
    rows: list[dict],
    seed: int = 13,
    labels: Iterable[str] | None = None,
) -> list[ConditionMetrics]:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    summaries = []
    for condition, condition_rows in sorted(by_condition.items()):
        n = len(condition_rows)
        correct = sum(row["correct"] for row in condition_rows)
        parsed = sum(row["pred_label"] is not None for row in condition_rows)
        ci_low, ci_high = bootstrap_accuracy_ci(condition_rows, seed=seed)
        summaries.append(
            ConditionMetrics(
                condition=condition,
                n=n,
                accuracy=correct / n if n else 0.0,
                parse_rate=parsed / n if n else 0.0,
                macro_f1=macro_f1(condition_rows, labels=labels),
                accuracy_ci_low=ci_low,
                accuracy_ci_high=ci_high,
            )
        )
    return summaries


def _correct_by_item(rows: list[dict], condition: str) -> dict[int, int]:
    values = {}
    for row in rows:
        if row["condition"] != condition:
            continue
        item_id = int(row["item_id"])
        if item_id in values:
            raise ValueError(f"Duplicate item_id {item_id} for condition {condition!r}")
        values[item_id] = int(row["correct"])
    return values


def paired_accuracy_deltas(
    rows: list[dict],
    baselines: Iterable[str],
    seed: int = 13,
    reps: int = 2000,
) -> list[PairedAccuracyDelta]:
    """Compare each non-baseline condition against each requested baseline."""

    baseline_names = tuple(baselines)
    conditions = sorted({row["condition"] for row in rows})
    treatment_conditions = [
        condition for condition in conditions if condition not in baseline_names
    ]
    results = []

    for baseline in baseline_names:
        baseline_values = _correct_by_item(rows, baseline)
        if not baseline_values:
            continue
        for condition in treatment_conditions:
            condition_values = _correct_by_item(rows, condition)
            if condition_values.keys() != baseline_values.keys():
                raise ValueError(
                    f"Paired comparison requires identical item IDs for "
                    f"{condition!r} and {baseline!r}"
                )

            item_ids = sorted(baseline_values)
            differences = [
                condition_values[item_id] - baseline_values[item_id]
                for item_id in item_ids
            ]
            n = len(differences)
            rng = random.Random(seed)
            bootstrap_deltas = []
            for _ in range(reps):
                bootstrap_deltas.append(
                    sum(differences[rng.randrange(n)] for _ in range(n)) / n
                )
            bootstrap_deltas.sort()

            results.append(
                PairedAccuracyDelta(
                    condition=condition,
                    baseline=baseline,
                    n=n,
                    accuracy_delta=sum(differences) / n,
                    accuracy_delta_ci_low=bootstrap_deltas[
                        int(0.025 * (reps - 1))
                    ],
                    accuracy_delta_ci_high=bootstrap_deltas[
                        int(0.975 * (reps - 1))
                    ],
                    wins=sum(difference > 0 for difference in differences),
                    ties=sum(difference == 0 for difference in differences),
                    losses=sum(difference < 0 for difference in differences),
                )
            )
    return results


def confusion_counts(rows: list[dict], limit: int = 50) -> list[dict]:
    counter = Counter(
        (row["gold_label"], row["pred_label"] or "<unparsed>")
        for row in rows
        if not row["correct"]
    )
    return [
        {"gold_label": gold, "pred_label": pred, "count": count}
        for (gold, pred), count in counter.most_common(limit)
    ]
