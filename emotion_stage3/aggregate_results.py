"""Aggregate locked Stage 3 runs with global confirmatory-family correction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from emotion_stage2.aggregate_results import _interaction_rows
from emotion_stage2.metrics import holm_adjust

from .io import write_csv
from .validate_run import validate_task_run


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def aggregate(run_dirs: list[Path], output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty {output_dir}")
    summaries = []
    paired = []
    run_metadata = []
    for run_dir in run_dirs:
        validation = validate_task_run(run_dir)
        if validation["status"] != "complete":
            raise ValueError(f"Run is incomplete: {run_dir}")
        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if (
            metadata["design"] != "confirmatory"
            or metadata["split"] != "test"
            or metadata["steering_lock"] is None
        ):
            raise ValueError(f"Aggregate only locked confirmatory tests: {run_dir}")
        identity = {
            "task": metadata["task"],
            "model": metadata["model"],
            "direction": metadata["direction"],
            "policy": metadata["policy"],
            "layers": metadata["layers"],
            "alpha": metadata["alpha"],
        }
        run_metadata.append(
            {
                "run_dir": str(run_dir.resolve()),
                **identity,
                "validation": validation,
            }
        )
        for row in _read_csv(run_dir / "summary.csv"):
            summaries.append({**identity, **row})
        for row in _read_csv(run_dir / "paired_deltas.csv"):
            paired.append(
                {
                    **identity,
                    "run_dir": str(run_dir.resolve()),
                    **row,
                }
            )

    confirmatory_indices = [
        index
        for index, row in enumerate(paired)
        if row["confirmatory_family"].lower() == "true"
    ]
    adjusted = holm_adjust(
        [float(paired[index]["p_value"]) for index in confirmatory_indices]
    )
    adjusted_by_index = dict(
        zip(confirmatory_indices, adjusted, strict=True)
    )
    for index, row in enumerate(paired):
        row["p_value_holm_global"] = adjusted_by_index.get(index, "")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "combined_summary.csv", summaries)
    write_csv(output_dir / "combined_paired_deltas.csv", paired)
    interactions = _interaction_rows(paired)
    write_csv(output_dir / "descriptive_interactions.csv", interactions)
    metadata = {
        "stage": (
            run_metadata[0]["validation"]["stage"]
            if run_metadata
            else 3
        ),
        "runs": run_metadata,
        "confirmatory_family_size": len(confirmatory_indices),
        "global_correction": "Holm-Bonferroni",
        "interaction_note": (
            "Interaction rows are descriptive differences between paired "
            "condition deltas, not standalone hypothesis tests."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "num_runs": len(run_dirs),
        "num_summary_rows": len(summaries),
        "num_paired_rows": len(paired),
        "confirmatory_family_size": len(confirmatory_indices),
        "num_globally_significant": sum(
            float(paired[index]["p_value_holm_global"]) < 0.05
            for index in confirmatory_indices
        ),
        "num_interaction_rows": len(interactions),
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
