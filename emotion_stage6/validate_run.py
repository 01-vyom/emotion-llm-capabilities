"""Validate a completed Stage 6 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotion_stage2.validate_run import validate as validate_generic_run
from emotion_stage3.io import sha256_file


def validate(run_dir: Path) -> dict:
    metadata = json.loads(
        (run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("stage") != 6:
        raise ValueError("Not a Stage 6 run")
    if metadata.get("design") not in {
        "admission_locked_prompt_only",
        "admission_locked_prompt_plus_dose_sign",
    }:
        raise ValueError("Unexpected Stage 6 design")
    panel_lock_path = Path(metadata["panel_lock_path"])
    if sha256_file(panel_lock_path) != metadata["panel_lock_sha256"]:
        raise ValueError("Stage 6 panel lock hash does not match")
    task_lock = metadata["panel_lock"]["tasks"][metadata["task"]]
    if task_lock["dataset_sha256"] != metadata["dataset_sha256"]:
        raise ValueError("Stage 6 dataset hash does not match admission lock")
    if task_lock["admitted_signals"] != metadata["primary_hypothesis_signals"]:
        raise ValueError("Stage 6 admitted signals do not match panel lock")
    if (
        metadata["task"] == "refusal_drift"
        and metadata["engine"] == "transformers"
        and metadata["dose_grid"]
    ):
        vector_path = Path(metadata["vector_dir"]) / "vectors.npz"
        if sha256_file(vector_path) != metadata["vectors_sha256"]:
            raise ValueError("Stage 6 vector hash does not match")
        if metadata["direction"] != "diligence":
            raise ValueError("Unexpected Stage 6 steering direction")
        if metadata["policy"] != "last":
            raise ValueError("Unexpected Stage 6 pooling policy")
    if metadata["engine"] == "bedrock":
        lock_path = Path(metadata["frontier_lock_path"])
        if sha256_file(lock_path) != metadata["frontier_lock_sha256"]:
            raise ValueError("Stage 6 frontier lock hash does not match")
    if not (run_dir / "conditional_primary_rates.csv").exists():
        raise ValueError("Missing conditional_primary_rates.csv")
    result = validate_generic_run(run_dir)
    return {
        **result,
        "stage": 6,
        "engine": metadata["engine"],
        "admitted_signals": metadata["primary_hypothesis_signals"],
        "not_testable_signals": metadata["not_testable_signals"],
        "direction": metadata["direction"],
        "policy": metadata["policy"],
        "layers": metadata["layers"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.run_dir)
    (args.run_dir / "validation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
