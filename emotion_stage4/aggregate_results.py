"""Aggregate the preregistered Stage 4 behavioral and cost families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotion_stage2.metrics import holm_adjust, paired_delta_dicts
from emotion_stage3.io import read_jsonl, write_csv

from .validate_run import validate


PROMPT_DOSE_CONDITIONS = (
    "stage4_calm_low__none",
    "stage4_calm_medium__none",
    "stage4_calm_high__none",
    "stage4_desperate_low__none",
    "stage4_desperate_medium__none",
    "stage4_desperate_high__none",
)
STEERING_COMPARISONS = (
    ("neutral_empty__positive", "neutral_empty__random"),
    ("neutral_empty__negative", "neutral_empty__random"),
    ("stage4_calm_high__positive", "stage4_calm_high__random"),
    ("stage4_desperate_high__negative", "stage4_desperate_high__random"),
)
PROMPT_BASELINE = "neutral_length_matched__none"


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
            sum(float(row["confidence"]) for row in bin_rows)
            / len(bin_rows)
        )
        value += len(bin_rows) / total * abs(accuracy - confidence)
    return value


def _mean(rows: list[dict], condition: str, signal: str) -> float:
    values = [
        float(row[signal])
        for row in rows
        if row["condition"] == condition
    ]
    if not values:
        raise ValueError(f"Missing {condition} rows for {signal}")
    return sum(values) / len(values)


def aggregate(run_dirs: list[Path], output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty {output_dir}")
    behavior_rows = []
    cost_rows = []
    ece_rows = []
    intensity_rows = []
    interaction_rows = []
    run_metadata = []

    for run_dir in run_dirs:
        validation = validate(run_dir)
        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata["split"] != "test":
            raise ValueError(f"Stage 4 aggregate requires test runs: {run_dir}")
        predictions = read_jsonl(run_dir / "predictions.jsonl")
        task = metadata["task"]
        seed = int(metadata["seed"])
        reps = int(metadata["bootstrap_reps"])
        primary_signals = tuple(metadata["primary_hypothesis_signals"])
        run_metadata.append(
            {
                "run_dir": str(run_dir.resolve()),
                "task": task,
                "model": metadata["model"],
                "primary_signals": list(primary_signals),
                "validation": validation,
            }
        )

        comparison_specs = [
            (condition, PROMPT_BASELINE, "prompt_dose")
            for condition in PROMPT_DOSE_CONDITIONS
        ] + [
            (condition, baseline, "steering")
            for condition, baseline in STEERING_COMPARISONS
        ]
        for signal in primary_signals:
            for condition, baseline, family in comparison_specs:
                row = _comparison(
                    predictions,
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
                        **row,
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

        for signal in primary_signals:
            for prompt_family in ("calm", "desperate"):
                conditions = [
                    f"stage4_{prompt_family}_{level}__none"
                    for level in ("low", "medium", "high")
                ]
                means = [_mean(predictions, condition, signal) for condition in conditions]
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

        for signal in primary_signals:
            for prompt_condition, vector_state in (
                ("stage4_calm_high", "positive"),
                ("stage4_desperate_high", "negative"),
            ):
                prompted = _mean(
                    predictions,
                    f"{prompt_condition}__{vector_state}",
                    signal,
                ) - _mean(
                    predictions,
                    f"{prompt_condition}__random",
                    signal,
                )
                neutral = _mean(
                    predictions,
                    f"neutral_empty__{vector_state}",
                    signal,
                ) - _mean(
                    predictions,
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

    behavior_adjusted = holm_adjust(
        [float(row["p_value"]) for row in behavior_rows]
    )
    for row, adjusted in zip(behavior_rows, behavior_adjusted, strict=True):
        row["p_value_holm_global"] = adjusted
    cost_adjusted = holm_adjust([float(row["p_value"]) for row in cost_rows])
    for row, adjusted in zip(cost_rows, cost_adjusted, strict=True):
        row["p_value_holm_cost"] = adjusted

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "behavioral_comparisons.csv", behavior_rows)
    write_csv(output_dir / "cost_comparisons.csv", cost_rows)
    write_csv(output_dir / "intensity_curves.csv", intensity_rows)
    write_csv(output_dir / "descriptive_interactions.csv", interaction_rows)
    write_csv(output_dir / "calibration_ece.csv", ece_rows)
    metadata = {
        "stage": 4,
        "runs": run_metadata,
        "behavioral_family_size": len(behavior_rows),
        "cost_family_size": len(cost_rows),
        "global_correction": "Holm-Bonferroni within behavioral and cost families",
        "behavioral_comparison_contract": {
            "prompt_dose_conditions": list(PROMPT_DOSE_CONDITIONS),
            "prompt_baseline": PROMPT_BASELINE,
            "steering_comparisons": [
                {"condition": condition, "baseline": baseline}
                for condition, baseline in STEERING_COMPARISONS
            ],
        },
        "interaction_note": (
            "Prompt-by-vector interactions and intensity contrasts are descriptive."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "num_runs": len(run_dirs),
        "behavioral_family_size": len(behavior_rows),
        "cost_family_size": len(cost_rows),
        "num_behavioral_significant": sum(
            float(row["p_value_holm_global"]) < 0.05
            for row in behavior_rows
        ),
        "num_cost_significant": sum(
            float(row["p_value_holm_cost"]) < 0.05
            for row in cost_rows
        ),
        "num_intensity_rows": len(intensity_rows),
        "num_interaction_rows": len(interaction_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate(args.run_dirs, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
