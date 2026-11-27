"""Shared paired and three-control comparisons for Stage 6."""

from __future__ import annotations

from emotion_stage2.metrics import paired_delta_dicts


def paired_comparison(
    rows: list[dict],
    *,
    condition: str,
    baseline: str,
    signal: str,
    seed: int,
    reps: int,
) -> dict:
    selected = [
        row for row in rows if row["condition"] in {condition, baseline}
    ]
    results = paired_delta_dicts(
        selected,
        signal_keys=[signal],
        baselines=[baseline],
        primary_signals=[signal],
        seed=seed,
        reps=reps,
    )
    if len(results) != 1:
        raise ValueError(
            f"Expected one paired comparison for {condition} vs {baseline}"
        )
    return results[0]


def condition_mean(rows: list[dict], condition: str, signal: str) -> float:
    values = [
        float(row[signal])
        for row in rows
        if row["condition"] == condition
    ]
    if not values:
        raise ValueError(f"Missing rows for {condition}/{signal}")
    return sum(values) / len(values)


def joint_three_control(
    rows: list[dict],
    *,
    condition: str,
    baselines: tuple[str, str, str],
    signal: str,
    seed: int,
    reps: int,
) -> tuple[dict, list[dict]]:
    components = [
        paired_comparison(
            rows,
            condition=condition,
            baseline=baseline,
            signal=signal,
            seed=seed,
            reps=reps,
        )
        for baseline in baselines
    ]
    deltas = [float(row["delta"]) for row in components]
    direction_consistent = all(value > 0 for value in deltas) or all(
        value < 0 for value in deltas
    )
    summary = {
        "condition": condition,
        "signal": signal,
        "n": int(components[0]["n"]),
        "joint_p_value": max(float(row["p_value"]) for row in components),
        "direction_consistent": direction_consistent,
        "joint_direction": (
            "positive"
            if all(value > 0 for value in deltas)
            else "negative"
            if all(value < 0 for value in deltas)
            else "mixed"
        ),
        "delta_vs_no_steer": float(components[0]["delta"]),
        "delta_vs_random": float(components[1]["delta"]),
        "delta_vs_shuffled": float(components[2]["delta"]),
    }
    return summary, components
