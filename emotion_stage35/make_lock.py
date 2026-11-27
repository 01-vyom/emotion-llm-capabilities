"""Create the fixed-reference-dose Stage 3.5 steering lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotion_stage3.io import sha256_file


def make_lock(
    probe_dir: Path,
    vector_dir: Path,
    *,
    alpha: float,
    direction_priority: tuple[str, ...],
) -> dict:
    report = json.loads(
        (probe_dir / "probe_report.json").read_text(encoding="utf-8")
    )
    passing = [
        row
        for row in report["best_by_pair_and_policy"]
        if row["passes_gate"]
    ]
    selected = None
    for direction in direction_priority:
        candidates = [row for row in passing if row["pair"] == direction]
        if candidates:
            selected = max(
                candidates,
                key=lambda row: (
                    float(row["direction_auc"]),
                    float(row["balanced_accuracy"]),
                    -int(row["layer"]),
                ),
            )
            break
    if selected is None:
        raise ValueError(
            "No lexical-applied passing direction matches the locked priority "
            f"{direction_priority}"
        )
    vector_meta = json.loads(
        (vector_dir / "extraction_meta.json").read_text(encoding="utf-8")
    )
    if vector_meta["model"] != report["model"]:
        raise ValueError("Probe and vector models do not match")
    return {
        "version": 2,
        "stage": 3.5,
        "model": report["model"],
        "task": "ifeval",
        "direction": selected["pair"],
        "policy": selected["policy"],
        "vectors_sha256": sha256_file(vector_dir / "vectors.npz"),
        "vector_dir": str(vector_dir.resolve()),
        "layers": [int(selected["layer"])],
        "alpha": alpha,
        "selection_rule": (
            "prefer diligence, then curious_vs_bored; within the first passing "
            "direction choose the held-out probe row with highest direction AUC; "
            "lock alpha=0.5 before behavioral outcomes"
        ),
        "selected_probe_row": selected,
        "direction_priority": list(direction_priority),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--direction-priority",
        nargs="+",
        default=["diligence", "curious_vs_bored"],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    lock = make_lock(
        args.probe_dir,
        args.vector_dir,
        alpha=args.alpha,
        direction_priority=tuple(args.direction_priority),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
