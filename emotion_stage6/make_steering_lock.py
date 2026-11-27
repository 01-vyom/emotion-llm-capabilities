"""Create a fixed-layer Stage 6 steering lock from a held-out probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotion_stage3.io import sha256_file


def make_steering_lock(
    probe_dir: Path,
    vector_dir: Path,
    *,
    direction: str,
    policy: str,
    layer: int,
    alpha: float,
) -> dict:
    report = json.loads(
        (probe_dir / "probe_report.json").read_text(encoding="utf-8")
    )
    candidates = [
        row
        for row in report["best_by_pair_and_policy"]
        if row["pair"] == direction
        and row["policy"] == policy
        and int(row["layer"]) == layer
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one fixed probe row for {direction}/{policy}/layer {layer}"
        )
    selected = candidates[0]
    if not selected["passes_gate"]:
        raise ValueError("Fixed Stage 6 probe row did not pass the gate")
    vector_meta = json.loads(
        (vector_dir / "extraction_meta.json").read_text(encoding="utf-8")
    )
    if vector_meta["model"] != report["model"]:
        raise ValueError("Probe and vector models do not match")
    if layer not in vector_meta["layers"]:
        raise ValueError("Fixed layer is absent from the vector artifact")
    return {
        "version": 1,
        "stage": 6,
        "artifact": "fixed_relative_depth_steering_lock",
        "model": report["model"],
        "direction": direction,
        "policy": policy,
        "layers": [layer],
        "alpha": alpha,
        "vectors_sha256": sha256_file(vector_dir / "vectors.npz"),
        "vector_dir": str(vector_dir.resolve()),
        "selected_probe_row": selected,
        "selection_rule": (
            "diligence, last-token pooling, one preregistered layer nearest "
            "two-thirds decoder depth; no outcome-driven layer search"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--direction", default="diligence")
    parser.add_argument("--policy", default="last")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    lock = make_steering_lock(
        args.probe_dir,
        args.vector_dir,
        direction=args.direction,
        policy=args.policy,
        layer=args.layer,
        alpha=args.alpha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
