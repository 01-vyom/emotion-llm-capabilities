"""Generic Stage 2 summaries, paired deltas, and multiplicity correction."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class PairedDelta:
    condition: str
    baseline: str
    signal: str
    n: int
    delta: float
    ci_low: float
    ci_high: float
    wins: int
    ties: int
    losses: int
    p_value: float
    p_value_holm: float | None
    primary_family: bool


def bootstrap_mean_ci(
    values: list[float],
    seed: int = 13,
    reps: int = 2000,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(reps)
    ]
    means.sort()
    return (
        means[int(0.025 * (reps - 1))],
        means[int(0.975 * (reps - 1))],
    )


def summarize_signals(
    rows: list[dict],
    signal_keys: Iterable[str],
    seed: int = 13,
    reps: int = 2000,
) -> list[dict]:
    conditions = sorted({row["condition"] for row in rows})
    summaries = []
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        summary: dict[str, object] = {
            "condition": condition,
            "n": len(condition_rows),
        }
        for signal in signal_keys:
            values = [float(row[signal]) for row in condition_rows]
            mean = sum(values) / len(values) if values else 0.0
            ci_low, ci_high = bootstrap_mean_ci(
                values,
                seed=seed,
                reps=reps,
            )
            summary[f"{signal}_mean"] = mean
            summary[f"{signal}_ci_low"] = ci_low
            summary[f"{signal}_ci_high"] = ci_high
        summaries.append(summary)
    return summaries


def _values_by_item(
    rows: list[dict],
    condition: str,
    signal: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in rows:
        if row["condition"] != condition:
            continue
        item_id = int(row["item_id"])
        if item_id in values:
            raise ValueError(f"Duplicate item_id {item_id} for condition {condition!r}")
        values[item_id] = float(row[signal])
    return values


def _two_sided_sign_test(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2 * probability)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def paired_deltas(
    rows: list[dict],
    signal_keys: Iterable[str],
    baselines: Iterable[str],
    primary_signals: Iterable[str],
    seed: int = 13,
    reps: int = 2000,
) -> list[PairedDelta]:
    baseline_names = tuple(baselines)
    primary_names = set(primary_signals)
    conditions = sorted({row["condition"] for row in rows})
    treatments = [
        condition for condition in conditions if condition not in baseline_names
    ]
    pending: list[dict] = []

    for signal in signal_keys:
        for baseline in baseline_names:
            baseline_values = _values_by_item(rows, baseline, signal)
            if not baseline_values:
                continue
            for condition in treatments:
                condition_values = _values_by_item(rows, condition, signal)
                if condition_values.keys() != baseline_values.keys():
                    raise ValueError(
                        "Paired comparison requires identical item IDs for "
                        f"{condition!r} and {baseline!r}"
                    )
                item_ids = sorted(baseline_values)
                differences = [
                    condition_values[item_id] - baseline_values[item_id]
                    for item_id in item_ids
                ]
                n = len(differences)
                rng = random.Random(seed)
                bootstrapped = [
                    sum(differences[rng.randrange(n)] for _ in range(n)) / n
                    for _ in range(reps)
                ]
                bootstrapped.sort()
                wins = sum(value > 0 for value in differences)
                ties = sum(value == 0 for value in differences)
                losses = sum(value < 0 for value in differences)
                pending.append(
                    {
                        "condition": condition,
                        "baseline": baseline,
                        "signal": signal,
                        "n": n,
                        "delta": sum(differences) / n,
                        "ci_low": bootstrapped[int(0.025 * (reps - 1))],
                        "ci_high": bootstrapped[int(0.975 * (reps - 1))],
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "p_value": _two_sided_sign_test(wins, losses),
                        "primary_family": signal in primary_names,
                    }
                )

    primary_indices = [
        index for index, row in enumerate(pending) if row["primary_family"]
    ]
    adjusted = holm_adjust([pending[index]["p_value"] for index in primary_indices])
    adjusted_by_index = dict(zip(primary_indices, adjusted, strict=True))

    results = []
    for index, row in enumerate(pending):
        results.append(
            PairedDelta(
                **row,
                p_value_holm=adjusted_by_index.get(index),
            )
        )
    return results


def paired_delta_dicts(*args, **kwargs) -> list[dict]:
    return [asdict(row) for row in paired_deltas(*args, **kwargs)]
