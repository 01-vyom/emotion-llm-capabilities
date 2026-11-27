"""Summarize Stage 3.5 dose curves, coherence, and layer placement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from emotion_stage3.io import write_csv
from emotion_stage3.validate_run import validate_task_run


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _alpha(condition: str) -> float:
    return float(condition.removeprefix("positive_a"))


def _run_rows(run_dir: Path) -> tuple[list[dict], dict]:
    validation = validate_task_run(run_dir)
    metadata = json.loads(
        (run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata["stage"] != 3.5 or metadata["design"] != "calibration":
        raise ValueError(f"Not a Stage 3.5 calibration run: {run_dir}")
    summary = {row["condition"]: row for row in _read_csv(run_dir / "summary.csv")}
    coherence_path = run_dir / "coherence_summary.csv"
    coherence = (
        {row["condition"]: row for row in _read_csv(coherence_path)}
        if coherence_path.exists()
        else {}
    )
    primary = metadata["primary_hypothesis_signals"][0]
    baseline = summary["no_steer"]
    rows = []
    for condition, row in summary.items():
        if not condition.startswith("positive_a"):
            continue
        alpha = _alpha(condition)
        random_condition = f"random_a{alpha:+.4f}"
        random_row = summary[random_condition]
        coherence_row = coherence.get(condition, {})
        rows.append(
            {
                "run_dir": str(run_dir.resolve()),
                "model": metadata["model"],
                "task": metadata["task"],
                "direction": metadata["direction"],
                "policy": metadata["policy"],
                "layers": metadata["layers"],
                "alpha": alpha,
                "primary_signal": primary,
                "primary_mean": float(row[f"{primary}_mean"]),
                "delta_vs_no_steer": (
                    float(row[f"{primary}_mean"])
                    - float(baseline[f"{primary}_mean"])
                ),
                "delta_vs_random": (
                    float(row[f"{primary}_mean"])
                    - float(random_row[f"{primary}_mean"])
                ),
                "parse_success": float(row["parse_success_mean"]),
                "output_truncated": float(row["output_truncated_mean"]),
                "output_tokens": float(row["output_tokens_mean"]),
                "conditional_perplexity": (
                    float(coherence_row["conditional_perplexity_mean"])
                    if coherence_row
                    else ""
                ),
                "validation_status": validation["status"],
            }
        )
    return rows, metadata


def _monotone(values: list[float]) -> bool:
    return all(
        current >= previous
        for previous, current in zip(values, values[1:], strict=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dose-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--layer-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--reference-alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {args.output_dir}")
    dose_rows = []
    dose_metadata = []
    for run_dir in args.dose_runs:
        rows, metadata = _run_rows(run_dir)
        dose_rows.extend(rows)
        dose_metadata.append(metadata)
    layer_rows = []
    layer_metadata = []
    for run_dir in args.layer_runs:
        rows, metadata = _run_rows(run_dir)
        matching = [
            row
            for row in rows
            if abs(float(row["alpha"]) - args.reference_alpha) < 1e-9
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Layer run {run_dir} lacks alpha {args.reference_alpha}"
            )
        layer_rows.extend(matching)
        layer_metadata.append(metadata)

    dose_reports = []
    for run_dir in args.dose_runs:
        rows = sorted(
            (
                row
                for row in dose_rows
                if row["run_dir"] == str(run_dir.resolve())
            ),
            key=lambda row: float(row["alpha"]),
        )
        baseline_ppl = ""
        coherence_path = run_dir / "coherence_summary.csv"
        if coherence_path.exists():
            coherence = {
                row["condition"]: row
                for row in _read_csv(coherence_path)
            }
            baseline_ppl = float(
                coherence["no_steer"]["conditional_perplexity_mean"]
            )
        wall = None
        for row in rows:
            incoherent = (
                float(row["parse_success"]) < 0.98
                or float(row["output_truncated"]) > 0.02
                or (
                    baseline_ppl != ""
                    and row["conditional_perplexity"] != ""
                    and float(row["conditional_perplexity"]) > 2 * baseline_ppl
                )
            )
            if incoherent:
                wall = row["alpha"]
                break
        reference = next(
            (
                row
                for row in rows
                if abs(float(row["alpha"]) - args.reference_alpha) < 1e-9
            ),
            None,
        )
        dose_reports.append(
            {
                "run_dir": str(run_dir.resolve()),
                "layers": rows[0]["layers"],
                "dose_monotone_vs_no_steer": _monotone(
                    [float(row["delta_vs_no_steer"]) for row in rows]
                ),
                "dose_monotone_vs_random": _monotone(
                    [float(row["delta_vs_random"]) for row in rows]
                ),
                "coherence_wall_alpha": wall,
                "reference_alpha": args.reference_alpha,
                "reference_delta_vs_no_steer": (
                    reference["delta_vs_no_steer"] if reference else None
                ),
                "reference_delta_vs_random": (
                    reference["delta_vs_random"] if reference else None
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "dose_response.csv", dose_rows)
    write_csv(args.output_dir / "layer_response.csv", layer_rows)
    write_csv(args.output_dir / "dose_report.csv", dose_reports)
    report = {
        "stage": 3.5,
        "reference_alpha": args.reference_alpha,
        "dose_runs": [str(path.resolve()) for path in args.dose_runs],
        "layer_runs": [str(path.resolve()) for path in args.layer_runs],
        "dose_reports": dose_reports,
        "num_layer_rows": len(layer_rows),
        "dose_metadata": dose_metadata,
        "layer_metadata": layer_metadata,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
