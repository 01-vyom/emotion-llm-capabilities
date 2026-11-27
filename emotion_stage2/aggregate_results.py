"""Aggregate locked Stage 2 runs and apply global primary-family correction."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .metrics import holm_adjust
from .validate_run import validate


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _interaction_rows(paired_rows: list[dict]) -> list[dict]:
    output = []
    model_groups: dict[tuple, list[dict]] = defaultdict(list)
    task_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in paired_rows:
        model_groups[
            (
                row["task"],
                row["condition"],
                row["baseline"],
                row["signal"],
            )
        ].append(row)
        task_groups[
            (
                row["model"],
                row["condition"],
                row["baseline"],
                row["signal"],
            )
        ].append(row)

    for key, group in sorted(model_groups.items()):
        if len({row["model"] for row in group}) < 2:
            continue
        task, condition, baseline, signal = key
        ordered = sorted(group, key=lambda row: row["model"])
        low, high = ordered[0], ordered[-1]
        output.append(
            {
                "interaction": "model_x_condition",
                "task": task,
                "model": f"{low['model']} vs {high['model']}",
                "condition": condition,
                "baseline": baseline,
                "signal": signal,
                "first_delta": low["delta"],
                "second_delta": high["delta"],
                "delta_difference": float(high["delta"]) - float(low["delta"]),
            }
        )

    for key, group in sorted(task_groups.items()):
        if len({row["task"] for row in group}) < 2:
            continue
        model, condition, baseline, signal = key
        ordered = sorted(group, key=lambda row: row["task"])
        low, high = ordered[0], ordered[-1]
        output.append(
            {
                "interaction": "task_x_condition",
                "task": f"{low['task']} vs {high['task']}",
                "model": model,
                "condition": condition,
                "baseline": baseline,
                "signal": signal,
                "first_delta": low["delta"],
                "second_delta": high["delta"],
                "delta_difference": float(high["delta"]) - float(low["delta"]),
            }
        )
    return output


def aggregate(run_dirs: list[Path], output_dir: Path) -> dict:
    summaries = []
    paired = []
    run_metadata = []
    for run_dir in run_dirs:
        validation = validate(run_dir)
        if validation["status"] != "complete":
            raise ValueError(f"Run is not complete: {run_dir}")
        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata["split"] != "test" or metadata["prompt_lock"] is None:
            raise ValueError(f"Aggregate only locked test runs: {run_dir}")
        run_metadata.append(
            {
                "run_dir": str(run_dir.resolve()),
                "task": metadata["task"],
                "model": metadata["model"],
            }
        )
        for row in _read_csv(run_dir / "summary.csv"):
            summaries.append(
                {
                    "task": metadata["task"],
                    "model": metadata["model"],
                    **row,
                }
            )
        for row in _read_csv(run_dir / "paired_deltas.csv"):
            paired.append(
                {
                    "task": metadata["task"],
                    "model": metadata["model"],
                    "run_dir": str(run_dir.resolve()),
                    **row,
                }
            )

    primary_indices = [
        index
        for index, row in enumerate(paired)
        if row["primary_family"].lower() == "true"
    ]
    globally_adjusted = holm_adjust(
        [float(paired[index]["p_value"]) for index in primary_indices]
    )
    adjusted_by_index = dict(zip(primary_indices, globally_adjusted, strict=True))
    for index, row in enumerate(paired):
        row["p_value_holm_global"] = adjusted_by_index.get(index, "")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "combined_summary.csv", summaries)
    _write_csv(output_dir / "combined_paired_deltas.csv", paired)
    interactions = _interaction_rows(paired)
    _write_csv(output_dir / "descriptive_interactions.csv", interactions)
    metadata = {
        "runs": run_metadata,
        "primary_family_size": len(primary_indices),
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
        "primary_family_size": len(primary_indices),
        "num_interaction_rows": len(interactions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.run_dirs, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
