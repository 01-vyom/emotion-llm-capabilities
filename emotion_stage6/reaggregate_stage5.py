"""Re-aggregate Stage 5 behavioral effects under the three-control rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotion_stage2.metrics import holm_adjust
from emotion_stage3.io import read_jsonl, write_csv
from emotion_stage4.aggregate_results import (
    PROMPT_BASELINE,
    PROMPT_DOSE_CONDITIONS,
)
from emotion_stage5.tasks import eligible_for_signal
from emotion_stage5.validate_run import validate

from .analysis import condition_mean, joint_three_control, paired_comparison


STEERING_CANDIDATES = (
    ("neutral_empty__positive", "neutral_empty"),
    ("neutral_empty__negative", "neutral_empty"),
    ("stage4_calm_high__positive", "stage4_calm_high"),
    ("stage4_desperate_high__negative", "stage4_desperate_high"),
)


def reaggregate(run_dirs: list[Path], output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty {output_dir}")
    candidates = []
    components = []
    means = []
    run_records = []
    panel_hash = None

    for run_dir in run_dirs:
        validation = validate(run_dir)
        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if panel_hash is None:
            panel_hash = metadata["panel_lock_sha256"]
        elif panel_hash != metadata["panel_lock_sha256"]:
            raise ValueError("Stage 5 runs use different panel locks")
        task = metadata["task"]
        rows = read_jsonl(run_dir / "predictions.jsonl")
        seed = int(metadata["seed"])
        reps = int(metadata["bootstrap_reps"])
        run_records.append(
            {
                "task": task,
                "run_dir": str(run_dir.resolve()),
                "validation": validation,
            }
        )
        for signal in metadata["primary_hypothesis_signals"]:
            eligible = [
                row
                for row in rows
                if eligible_for_signal(task, signal, row)
            ]
            for condition in metadata["arms"]:
                means.append(
                    {
                        "task": task,
                        "signal": signal,
                        "condition": condition,
                        "mean": condition_mean(eligible, condition, signal),
                    }
                )
            for condition in PROMPT_DOSE_CONDITIONS:
                comparison = paired_comparison(
                    eligible,
                    condition=condition,
                    baseline=PROMPT_BASELINE,
                    signal=signal,
                    seed=seed,
                    reps=reps,
                )
                candidates.append(
                    {
                        "task": task,
                        "signal": signal,
                        "candidate_type": "prompt",
                        "condition": condition,
                        "baseline": PROMPT_BASELINE,
                        "p_value_candidate": comparison["p_value"],
                        "direction_consistent": True,
                        "delta_vs_no_steer": comparison["delta"],
                        "delta_vs_random": "",
                        "delta_vs_shuffled": "",
                        "n": comparison["n"],
                    }
                )
                components.append(
                    {
                        "task": task,
                        "signal": signal,
                        "candidate_type": "prompt",
                        **comparison,
                    }
                )
            for condition, prompt_condition in STEERING_CANDIDATES:
                baselines = (
                    f"{prompt_condition}__none",
                    f"{prompt_condition}__random",
                    f"{prompt_condition}__shuffled",
                )
                joint, candidate_components = joint_three_control(
                    eligible,
                    condition=condition,
                    baselines=baselines,
                    signal=signal,
                    seed=seed,
                    reps=reps,
                )
                candidates.append(
                    {
                        "task": task,
                        "signal": signal,
                        "candidate_type": "steering",
                        "condition": condition,
                        "baseline": "|".join(baselines),
                        "p_value_candidate": joint["joint_p_value"],
                        "direction_consistent": joint["direction_consistent"],
                        "delta_vs_no_steer": joint["delta_vs_no_steer"],
                        "delta_vs_random": joint["delta_vs_random"],
                        "delta_vs_shuffled": joint["delta_vs_shuffled"],
                        "n": joint["n"],
                    }
                )
                for component in candidate_components:
                    components.append(
                        {
                            "task": task,
                            "signal": signal,
                            "candidate_type": "steering",
                            **component,
                        }
                    )

    adjusted = holm_adjust(
        [float(row["p_value_candidate"]) for row in candidates]
    )
    for row, value in zip(candidates, adjusted, strict=True):
        row["p_value_holm_global"] = value
        row["retained"] = (
            bool(row["direction_consistent"]) and float(value) < 0.05
        )
    retained = [row for row in candidates if row["retained"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "candidate_effects.csv", candidates)
    write_csv(output_dir / "component_comparisons.csv", components)
    write_csv(output_dir / "arm_means.csv", means)
    metadata = {
        "stage": 6,
        "artifact": "stage5_three_control_reanalysis",
        "source_stage": 5,
        "panel_lock_sha256": panel_hash,
        "runs": run_records,
        "family_size": len(candidates),
        "joint_test": (
            "intersection-union maximum component p-value; all three deltas "
            "must have one nonzero direction"
        ),
        "global_correction": "Holm-Bonferroni across candidate effects",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    summary = {
        "stage": 6,
        "artifact": "stage5_three_control_reanalysis",
        "family_size": len(candidates),
        "num_retained": len(retained),
        "retained": retained,
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
    print(json.dumps(reaggregate(args.run_dirs, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
