"""Validate Stage 3 vector, probe, or task-run artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from emotion_stage2.validate_run import validate as validate_stage2_run

from .io import sha256_file


def validate_vectors(vector_dir: Path) -> dict:
    meta = json.loads(
        (vector_dir / "extraction_meta.json").read_text(encoding="utf-8")
    )
    vectors = np.load(vector_dir / "vectors.npz")
    activations = np.load(vector_dir / "activations.npz")
    layers = vectors["layers"]
    pairs = vectors["pair_names"]
    expected_prefix = (len(pairs), len(layers), meta["hidden_size"])
    for policy in meta["position_policies"]:
        for suffix in ("raw", "unit", "shuffled_unit", "random_unit"):
            key = f"{policy}_{suffix}"
            if vectors[key].shape != expected_prefix:
                raise ValueError(
                    f"{key} has shape {vectors[key].shape}, expected "
                    f"{expected_prefix}"
                )
            if not np.isfinite(vectors[key]).all():
                raise ValueError(f"{key} contains non-finite values")
        for key in (
            f"{policy}_unit",
            f"{policy}_shuffled_unit",
            f"{policy}_random_unit",
        ):
            if not np.allclose(
                np.linalg.norm(vectors[key], axis=-1),
                1.0,
                atol=1e-4,
            ):
                raise ValueError(f"{key} contains non-unit vectors")
        expected_activations = (
            meta["num_vignettes"],
            len(layers),
            meta["hidden_size"],
        )
        if activations[policy].shape != expected_activations:
            raise ValueError(
                f"{policy} activations have shape {activations[policy].shape}, "
                f"expected {expected_activations}"
            )
    if len(activations["vignette_ids"]) != meta["num_vignettes"]:
        raise ValueError("Activation vignette IDs do not match metadata")
    return {
        "status": "complete",
        "artifact": "activation_vectors",
        "stage": meta.get("stage", 3),
        "model": meta["model"],
        "num_vignettes": meta["num_vignettes"],
        "num_layers": len(layers),
        "num_pairs": len(pairs),
        "vectors_sha256": sha256_file(vector_dir / "vectors.npz"),
    }


def validate_probe(probe_dir: Path) -> dict:
    report = json.loads(
        (probe_dir / "probe_report.json").read_text(encoding="utf-8")
    )
    if (
        report["stage"] not in {3, 3.5}
        or report["artifact"] != "probe_gate"
    ):
        raise ValueError("Not a Stage 3 or Stage 3.5 probe report")
    if not (probe_dir / "layer_profile.csv").exists():
        raise ValueError("Probe report is missing layer_profile.csv")
    for name in ("layer_profile.png", "layer_profile.svg"):
        if not (probe_dir / name).exists():
            raise ValueError(f"Probe report is missing {name}")
    for row in report["best_by_pair_and_policy"]:
        for key in (
            "direction_auc",
            "balanced_accuracy",
            "lexical_auc",
            "shuffled_auc_oriented",
            "random_auc_oriented",
            "logistic_probe_auc",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Probe report has non-finite {key}")
    return {
        "status": "complete",
        "artifact": "probe_gate",
        "stage": report["stage"],
        "model": report["model"],
        "gate_passed": report["gate_passed"],
        "gate_passed_with_abstain": report.get(
            "gate_passed_with_abstain",
            report["gate_passed"],
        ),
        "primary_diligence_passed": report["primary_diligence_passed"],
        "num_passing_directions": len(report["passing_directions"]),
        "num_passing_directions_lexical_applied": len(
            report.get(
                "passing_directions_lexical_applied",
                report["passing_directions"],
            )
        ),
        "num_passing_directions_with_abstain": len(
            report.get("passing_directions_with_abstain", [])
        ),
    }


def validate_task_run(run_dir: Path) -> dict:
    metadata = json.loads(
        (run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("stage") not in {3, 3.5}:
        raise ValueError("Not a Stage 3 or Stage 3.5 task run")
    if (
        metadata["design"] == "confirmatory"
        and metadata["split"] == "test"
        and metadata["steering_lock"] is None
    ):
        raise ValueError("Held-out confirmatory run has no steering lock")
    vector_dir = metadata.get("vector_dir")
    if vector_dir is not None:
        vector_path = Path(vector_dir) / "vectors.npz"
        if not vector_path.exists():
            raise ValueError(f"Missing vector artifact {vector_path}")
        if sha256_file(vector_path) != metadata["vectors_sha256"]:
            raise ValueError("Task run vector hash does not match current artifact")
    result = validate_stage2_run(run_dir)
    return {
        **result,
        "stage": metadata["stage"],
        "design": metadata["design"],
        "direction": metadata["direction"],
        "policy": metadata["policy"],
        "layers": metadata["layers"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vector-dir", type=Path)
    group.add_argument("--probe-dir", type=Path)
    group.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vector_dir is not None:
        result = validate_vectors(args.vector_dir)
        output = args.vector_dir / "validation.json"
    elif args.probe_dir is not None:
        result = validate_probe(args.probe_dir)
        output = args.probe_dir / "validation.json"
    else:
        result = validate_task_run(args.run_dir)
        output = args.run_dir / "validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
