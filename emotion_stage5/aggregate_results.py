"""Aggregate admitted Stage 5 DVs with global behavioral and cost correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotion_stage2.metrics import holm_adjust, paired_delta_dicts
from emotion_stage3.io import read_jsonl, write_csv
from emotion_stage4.aggregate_results import (
    PROMPT_BASELINE,
    PROMPT_DOSE_CONDITIONS,
    STEERING_COMPARISONS,
)

from .tasks import eligible_for_signal
from .validate_run import validate


def _comparison(
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


def _mean(rows: list[dict], condition: str, signal: str) -> float:
    values = [
        float(row[signal])
        for row in rows
        if row["condition"] == condition
    ]
    if not values:
        raise ValueError(f"Missing {condition} rows for {signal}")
    return sum(values) / len(values)


def _ece(rows: list[dict], bins: int = 10) -> float:
    parsed = [row for row in rows if float(row["parse_success"]) == 1.0]
    if not parsed:
        return 1.0
    total = len(parsed)
    value = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        bin_rows = [
            row
            for row in parsed
            if low <= float(row["confidence"]) < high
            or (
                bin_index == bins - 1
                and float(row["confidence"]) == 1.0
            )
        ]
        if not bin_rows:
            continue
        accuracy = sum(float(row["correct"]) for row in bin_rows) / len(bin_rows)
        confidence = (
            sum(float(row["confidence"]) for row in bin_rows) / len(bin_rows)
        )
        value += len(bin_rows) / total * abs(accuracy - confidence)
    return value


def aggregate(run_dirs: list[Path], output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty {output_dir}")
    behavior_rows = []
    cost_rows = []
    intensity_rows = []
    interaction_rows = []
    ece_rows = []
    run_metadata = []
    panel_lock = None
    panel_lock_hash = None
    observed_tasks = set()

    comparison_specs = [
        (condition, PROMPT_BASELINE, "prompt_dose")
        for condition in PROMPT_DOSE_CONDITIONS
    ] + [
        (condition, baseline, "steering")
        for condition, baseline in STEERING_COMPARISONS
    ]

    for run_dir in run_dirs:
        validation = validate(run_dir)
        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if not metadata.get("confirmatory"):
            raise ValueError(f"Stage 5 aggregate cannot consume smoke run: {run_dir}")
        task = metadata["task"]
        if task in observed_tasks:
            raise ValueError(f"Duplicate Stage 5 run for task {task}")
        observed_tasks.add(task)
        if panel_lock is None:
            panel_lock = metadata["panel_lock"]
            panel_lock_hash = metadata["panel_lock_sha256"]
        elif metadata["panel_lock_sha256"] != panel_lock_hash:
            raise ValueError("Stage 5 runs use different panel locks")
        predictions = read_jsonl(run_dir / "predictions.jsonl")
        seed = int(metadata["seed"])
        reps = int(metadata["bootstrap_reps"])
        admitted_signals = tuple(metadata["primary_hypothesis_signals"])
        run_metadata.append(
            {
                "run_dir": str(run_dir.resolve()),
                "task": task,
                "model": metadata["model"],
                "admitted_signals": list(admitted_signals),
                "not_testable_signals": metadata["not_testable_signals"],
                "validation": validation,
            }
        )

        for signal in admitted_signals:
            eligible_rows = [
                row
                for row in predictions
                if eligible_for_signal(task, signal, row)
            ]
            for condition, baseline, family in comparison_specs:
                row = _comparison(
                    eligible_rows,
                    condition=condition,
                    baseline=baseline,
                    signal=signal,
                    seed=seed,
                    reps=reps,
                )
                behavior_rows.append(
                    {
                        "task": task,
                        "run_dir": str(run_dir.resolve()),
                        "comparison_family": family,
                        "eligible_n": sum(
                            candidate["condition"] == condition
                            for candidate in eligible_rows
                        ),
                        **row,
                    }
                )
            for prompt_family in ("calm", "desperate"):
                conditions = [
                    f"stage4_{prompt_family}_{level}__none"
                    for level in ("low", "medium", "high")
                ]
                means = [
                    _mean(eligible_rows, condition, signal)
                    for condition in conditions
                ]
                intensity_rows.append(
                    {
                        "task": task,
                        "signal": signal,
                        "prompt_family": prompt_family,
                        "low_mean": means[0],
                        "medium_mean": means[1],
                        "high_mean": means[2],
                        "medium_vs_endpoint_average": (
                            means[1] - (means[0] + means[2]) / 2
                        ),
                    }
                )
            for prompt_condition, vector_state in (
                ("stage4_calm_high", "positive"),
                ("stage4_desperate_high", "negative"),
            ):
                prompted = _mean(
                    eligible_rows,
                    f"{prompt_condition}__{vector_state}",
                    signal,
                ) - _mean(
                    eligible_rows,
                    f"{prompt_condition}__random",
                    signal,
                )
                neutral = _mean(
                    eligible_rows,
                    f"neutral_empty__{vector_state}",
                    signal,
                ) - _mean(
                    eligible_rows,
                    "neutral_empty__random",
                    signal,
                )
                interaction_rows.append(
                    {
                        "task": task,
                        "signal": signal,
                        "prompt_condition": prompt_condition,
                        "vector_state": vector_state,
                        "prompted_vector_delta": prompted,
                        "neutral_vector_delta": neutral,
                        "interaction_delta": prompted - neutral,
                    }
                )

        for condition, baseline, family in comparison_specs:
            row = _comparison(
                predictions,
                condition=condition,
                baseline=baseline,
                signal="output_tokens",
                seed=seed,
                reps=reps,
            )
            cost_rows.append(
                {
                    "task": task,
                    "run_dir": str(run_dir.resolve()),
                    "comparison_family": family,
                    **row,
                }
            )

        if task == "calibration":
            for condition in metadata["arms"]:
                condition_rows = [
                    row for row in predictions if row["condition"] == condition
                ]
                ece_rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "n": len(condition_rows),
                        "ece_10_bin": _ece(condition_rows),
                    }
                )

    if panel_lock is None:
        raise ValueError("At least one Stage 5 run is required")
    expected_tasks = set(panel_lock["admitted_tasks"])
    if observed_tasks != expected_tasks:
        raise ValueError(
            f"Stage 5 run task mismatch; expected={sorted(expected_tasks)}, "
            f"observed={sorted(observed_tasks)}"
        )

    behavior_adjusted = holm_adjust(
        [float(row["p_value"]) for row in behavior_rows]
    )
    for row, adjusted in zip(behavior_rows, behavior_adjusted, strict=True):
        row["p_value_holm_global"] = adjusted
    cost_adjusted = holm_adjust([float(row["p_value"]) for row in cost_rows])
    for row, adjusted in zip(cost_rows, cost_adjusted, strict=True):
        row["p_value_holm_cost"] = adjusted

    admission_rows = []
    for task, task_lock in panel_lock["tasks"].items():
        for rate in task_lock["rates"]:
            admission_rows.append(
                {
                    **rate,
                    "status": (
                        "admitted"
                        if rate["signal"] in task_lock["admitted_signals"]
                        else "not_testable"
                    ),
                }
            )
    not_testable_rows = [
        row for row in admission_rows if row["status"] == "not_testable"
    ]
    significant_behavioral = [
        row
        for row in behavior_rows
        if float(row["p_value_holm_global"]) < 0.05
    ]
    significant_cost = [
        row for row in cost_rows if float(row["p_value_holm_cost"]) < 0.05
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "admission_status.csv", admission_rows)
    write_csv(output_dir / "not_testable.csv", not_testable_rows)
    write_csv(output_dir / "behavioral_comparisons.csv", behavior_rows)
    write_csv(output_dir / "cost_comparisons.csv", cost_rows)
    write_csv(output_dir / "intensity_curves.csv", intensity_rows)
    write_csv(output_dir / "descriptive_interactions.csv", interaction_rows)
    write_csv(output_dir / "calibration_ece.csv", ece_rows)
    metadata = {
        "stage": 5,
        "model": panel_lock["model"],
        "panel_lock": panel_lock,
        "panel_lock_sha256": panel_lock_hash,
        "runs": run_metadata,
        "admitted_signal_count": len(panel_lock["admitted_signals"]),
        "not_testable_signal_count": len(panel_lock["not_testable_signals"]),
        "behavioral_family_size": len(behavior_rows),
        "cost_family_size": len(cost_rows),
        "global_correction": (
            "Holm-Bonferroni within admitted behavioral and cost families"
        ),
        "interaction_note": (
            "Prompt-by-vector interactions and intensity contrasts are descriptive."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    summary = {
        "stage": 5,
        "model": panel_lock["model"],
        "num_runs": len(run_dirs),
        "admitted_signals": panel_lock["admitted_signals"],
        "not_testable_signals": panel_lock["not_testable_signals"],
        "behavioral_family_size": len(behavior_rows),
        "num_behavioral_significant": len(significant_behavioral),
        "cost_family_size": len(cost_rows),
        "num_cost_significant": len(significant_cost),
        "significant_behavioral": significant_behavioral,
        "significant_cost": significant_cost,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.run_dirs, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
