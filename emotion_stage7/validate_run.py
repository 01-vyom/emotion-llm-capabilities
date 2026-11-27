"""Validate a completed Stage 7 treatment run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotion_stage2.validate_run import validate as validate_generic_run
from emotion_stage3.io import sha256_file

from .constants import (
    IFEVAL_MAX_NEW_TOKENS,
    IFEVAL_RUNTIME_AMENDMENT_SHA256,
    PREREGISTRATION_SHA256,
)


DESIGNS = {
    "difficulty_locked_prompt_only",
    "difficulty_locked_prompt_plus_dose_sign",
    "full_ladder_ifeval_prompt_plus_steering",
    "full_ladder_ifeval_prompt_only",
}


def validate(run_dir: Path) -> dict:
    metadata = json.loads(
        (run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("stage") != 7 or metadata.get("design") not in DESIGNS:
        raise ValueError("Not a recognized Stage 7 run")
    if metadata.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("Stage 7 preregistration hash does not match")
    if metadata["task"] == "ifeval":
        if (
            metadata.get("ifeval_runtime_amendment_sha256")
            != IFEVAL_RUNTIME_AMENDMENT_SHA256
        ):
            raise ValueError("Stage 7 IFEval runtime amendment hash does not match")
        if metadata.get("max_new_tokens") != IFEVAL_MAX_NEW_TOKENS:
            raise ValueError("Stage 7 IFEval generation cap does not match")
    if metadata["task"] in {"sycophancy", "refusal_drift"}:
        panel_path = Path(metadata["panel_lock_path"])
        if sha256_file(panel_path) != metadata["panel_lock_sha256"]:
            raise ValueError("Stage 7 panel lock hash does not match")
        task_lock = metadata["panel_lock"]["tasks"][metadata["task"]]
        if task_lock["dataset_sha256"] != metadata["dataset_sha256"]:
            raise ValueError("Stage 7 selected dataset hash does not match")
        if task_lock["admitted_signals"] != metadata["primary_hypothesis_signals"]:
            raise ValueError("Stage 7 admitted signals do not match panel lock")
        difficulty_path = Path(metadata["difficulty_lock_path"])
        if sha256_file(difficulty_path) != metadata["difficulty_lock_sha256"]:
            raise ValueError("Stage 7 difficulty lock hash does not match")
        if not (run_dir / "conditional_primary_rates.csv").exists():
            raise ValueError("Missing conditional_primary_rates.csv")
    has_steering = any(
        arm not in {
            "neutral_empty__none",
            "neutral_length_matched__none",
            "stage4_calm_low__none",
            "stage4_calm_medium__none",
            "stage4_calm_high__none",
            "stage4_desperate_low__none",
            "stage4_desperate_medium__none",
            "stage4_desperate_high__none",
        }
        for arm in metadata["arms"]
    )
    if has_steering:
        vector_path = Path(metadata["vector_dir"]) / "vectors.npz"
        if sha256_file(vector_path) != metadata["vectors_sha256"]:
            raise ValueError("Stage 7 vector hash does not match")
        if metadata["direction"] != "diligence" or metadata["policy"] != "last":
            raise ValueError("Unexpected Stage 7 steering configuration")
    if metadata["engine"] == "bedrock":
        frontier_path = Path(metadata["frontier_lock_path"])
        if sha256_file(frontier_path) != metadata["frontier_lock_sha256"]:
            raise ValueError("Stage 7 frontier lock hash does not match")
    result = validate_generic_run(run_dir)
    return {
        **result,
        "stage": 7,
        "engine": metadata["engine"],
        "admitted_signals": metadata["primary_hypothesis_signals"],
        "not_testable_signals": metadata.get("not_testable_signals", []),
        "direction": metadata.get("direction"),
        "policy": metadata.get("policy"),
        "layers": metadata.get("layers", []),
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
