"""Select a coherent Stage 3 alpha/layer setting from dev calibration runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .io import sha256_file
from .validate_run import validate_task_run


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select(run_dirs: list[Path]) -> dict:
    candidates = []
    shared: dict | None = None
    for run_dir in run_dirs:
        validation = validate_task_run(run_dir)
        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata["design"] != "calibration" or metadata["split"] != "dev":
            raise ValueError(f"Not a dev calibration run: {run_dir}")
        identity = {
            "model": metadata["model"],
            "task": metadata["task"],
            "direction": metadata["direction"],
            "policy": metadata["policy"],
            "vectors_sha256": metadata["vectors_sha256"],
        }
        if shared is None:
            shared = identity
        elif identity != shared:
            raise ValueError("Calibration runs do not share one configuration")
        primary = metadata["primary_hypothesis_signals"][0]
        summary = {
            row["condition"]: row
            for row in _read_csv(run_dir / "summary.csv")
        }
        baseline = float(summary["no_steer"][f"{primary}_mean"])
        baseline_parse = float(summary["no_steer"]["parse_success_mean"])
        for arm, row in summary.items():
            if not arm.startswith("positive_a"):
                continue
            alpha_label = arm.removeprefix("positive_a")
            random_arm = f"random_a{alpha_label}"
            if random_arm not in summary:
                raise ValueError(f"Missing matched random arm {random_arm}")
            parse_success = float(row["parse_success_mean"])
            truncation = float(row["output_truncated_mean"])
            random_mean = float(summary[random_arm][f"{primary}_mean"])
            primary_mean = float(row[f"{primary}_mean"])
            coherent = (
                parse_success >= baseline_parse - 0.02
                and truncation <= 0.02
            )
            candidates.append(
                {
                    "run_dir": str(run_dir.resolve()),
                    "layers": metadata["layers"],
                    "alpha": float(alpha_label),
                    "primary_signal": primary,
                    "primary_mean": primary_mean,
                    "delta_vs_no_steer": primary_mean - baseline,
                    "delta_vs_random": primary_mean - random_mean,
                    "parse_success": parse_success,
                    "output_truncated": truncation,
                    "coherent": coherent,
                    "validation": validation,
                }
            )
    coherent = [candidate for candidate in candidates if candidate["coherent"]]
    if not coherent:
        raise ValueError("No coherent alpha/layer calibration candidate")
    selected = max(
        coherent,
        key=lambda row: (
            row["delta_vs_random"],
            row["delta_vs_no_steer"],
            -abs(row["alpha"]),
            -len(row["layers"]),
        ),
    )
    return {
        "version": 1,
        **shared,
        "vector_dir": json.loads(
            (Path(selected["run_dir"]) / "metadata.json").read_text(
                encoding="utf-8"
            )
        )["vector_dir"],
        "direction": shared["direction"],
        "policy": shared["policy"],
        "layers": selected["layers"],
        "alpha": selected["alpha"],
        "selection_rule": (
            "maximize dev primary delta vs matched random control, then "
            "no-steer delta, then prefer smaller alpha and fewer layers; "
            "require parse_success within 0.02 of baseline and truncation <= 0.02"
        ),
        "selected_candidate": selected,
        "all_candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    lock = select(args.run_dirs)
    vector_path = Path(lock["vector_dir"]) / "vectors.npz"
    if sha256_file(vector_path) != lock["vectors_sha256"]:
        raise RuntimeError("Selected vector hash changed during lock creation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
