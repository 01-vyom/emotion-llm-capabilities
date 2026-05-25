"""Metrics and confidence intervals for Stage 1 outputs."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionMetrics:
    condition: str
    n: int
    accuracy: float
    parse_rate: float
    macro_f1: float
    accuracy_ci_low: float
    accuracy_ci_high: float


def macro_f1(rows: list[dict]) -> float:
    labels = sorted({row["gold_label"] for row in rows} | {row["pred_label"] for row in rows if row["pred_label"]})
    if not labels:
        return 0.0

    f1s = []
    for label in labels:
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


def summarize(rows: list[dict], seed: int = 13) -> list[ConditionMetrics]:
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
                macro_f1=macro_f1(condition_rows),
                accuracy_ci_low=ci_low,
                accuracy_ci_high=ci_high,
            )
        )
    return summaries


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
