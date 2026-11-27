"""Validate Stage 2 prompt or completed-run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _assert_paired_items(rows: list[dict], arms: Iterable[str]) -> int:
    by_arm: dict[str, set[int]] = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["condition"] == arm]
        item_ids = [int(row["item_id"]) for row in arm_rows]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError(f"Duplicate item IDs in arm {arm!r}")
        by_arm[arm] = set(item_ids)
    item_sets = {tuple(sorted(item_ids)) for item_ids in by_arm.values()}
    if len(item_sets) != 1:
        raise ValueError("Experimental arms do not contain identical item IDs")
    return len(next(iter(item_sets), ()))


def validate(run_dir: Path) -> dict:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    prompts = _read_jsonl(run_dir / "prompts.jsonl")
    expected_prompts = int(metadata["num_prompts"])
    if len(prompts) != expected_prompts:
        raise ValueError(
            f"Expected {expected_prompts} prompts, found {len(prompts)}"
        )
    arms = metadata["arms"]
    items_per_arm = _assert_paired_items(prompts, arms)
    if items_per_arm != int(metadata["num_examples"]):
        raise ValueError("Prompt item count does not match metadata")

    if metadata["task"] == "ruler_niah":
        for row in prompts:
            if row["mode"] == "aggregate":
                document = row["prompt"].split("\n\n[QUERY]\n", maxsplit=1)[0]
                marker = row["aggregate_marker"]
                expected_count = int(row["gold"])
                if document.count(marker) != expected_count:
                    raise ValueError(
                        f"Aggregate marker {marker!r} occurs "
                        f"{document.count(marker)} times, expected {expected_count}"
                    )
                continue
            for target in row["gold"]:
                if row["prompt"].count(target) != 1:
                    raise ValueError(
                        f"Needle {target!r} does not occur exactly once in prompt"
                    )
    elif metadata["task"] == "ifeval":
        from instruction_following_eval.instructions_registry import INSTRUCTION_DICT

        unknown = sorted(
            {
                instruction_id
                for row in prompts
                for instruction_id in row["instruction_id_list"]
                if instruction_id not in INSTRUCTION_DICT
            }
        )
        if unknown:
            raise ValueError(f"Unknown IFEval instruction IDs: {unknown}")

    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        return {
            "status": "prompts_only",
            "task": metadata["task"],
            "model": metadata["model"],
            "num_arms": len(arms),
            "items_per_arm": items_per_arm,
            "num_prompts": len(prompts),
        }

    predictions = _read_jsonl(predictions_path)
    if len(predictions) != expected_prompts:
        raise ValueError(
            f"Expected {expected_prompts} predictions, found {len(predictions)}"
        )
    _assert_paired_items(predictions, arms)
    for row in predictions:
        for signal in metadata["signal_keys"]:
            value = float(row[signal])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {signal!r} for item {row['item_id']}")

    summary = _read_csv(run_dir / "summary.csv")
    if {row["condition"] for row in summary} != set(arms):
        raise ValueError("Summary arms do not match metadata")

    paired = _read_csv(run_dir / "paired_deltas.csv")
    baselines = metadata["paired_baselines"]
    expected_paired = (
        (len(arms) - len(baselines))
        * len(baselines)
        * len(metadata["signal_keys"])
    )
    if len(paired) != expected_paired:
        raise ValueError(
            f"Expected {expected_paired} paired rows, found {len(paired)}"
        )

    required_files = (
        "confusions.jsonl",
        "run_stats.json",
        "requirements.freeze.txt",
        "nvidia-smi.txt",
    )
    missing = [name for name in required_files if not (run_dir / name).exists()]
    if missing:
        raise ValueError(f"Missing required artifacts: {missing}")

    return {
        "status": "complete",
        "task": metadata["task"],
        "model": metadata["model"],
        "num_arms": len(arms),
        "items_per_arm": items_per_arm,
        "num_prompts": len(prompts),
        "num_predictions": len(predictions),
        "num_summary_rows": len(summary),
        "num_paired_rows": len(paired),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.run_dir)
    output = args.run_dir / "validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
