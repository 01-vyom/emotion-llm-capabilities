"""Validate a completed Stage 4 safety-panel run."""

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
    if metadata.get("stage") != 4:
        raise ValueError("Not a Stage 4 run")
    if metadata.get("design") != "crossed_prompt_by_steering":
        raise ValueError("Unexpected Stage 4 design")
    if metadata.get("steering_lock") is None:
        raise ValueError("Stage 4 run has no steering lock")
    vector_path = Path(metadata["vector_dir"]) / "vectors.npz"
    if not vector_path.exists():
        raise ValueError(f"Missing vector artifact {vector_path}")
    if sha256_file(vector_path) != metadata["vectors_sha256"]:
        raise ValueError("Stage 4 vector hash does not match")
    result = validate_generic_run(run_dir)
    return {
        **result,
        "stage": 4,
        "direction": metadata["direction"],
        "policy": metadata["policy"],
        "layers": metadata["layers"],
        "alpha": metadata["alpha"],
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
