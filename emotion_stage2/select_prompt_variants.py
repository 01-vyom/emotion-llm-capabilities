"""Select one Stage 2 affect-cue variant per condition from a dev run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def select_variants(dev_run: Path) -> dict:
    metadata = json.loads((dev_run / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("split") != "dev" or not metadata.get("all_variants"):
        raise ValueError("Prompt selection requires a dev run with --all-variants")
    primary_signals = metadata["primary_hypothesis_signals"]
    if len(primary_signals) != 1:
        raise ValueError("Prompt selection currently requires one primary signal")
    signal = primary_signals[0]
    metric = f"{signal}_mean"
    with (dev_run / "summary.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    selected: dict[str, int] = {}
    selected_scores: dict[str, float] = {}
    for condition in metadata["conditions"]:
        candidates = []
        for row in rows:
            arm = row["condition"]
            prefix = f"{condition}::v"
            if arm.startswith(prefix):
                variant = int(arm[len(prefix) :])
                candidates.append((float(row[metric]), -variant, variant))
        if not candidates:
            raise ValueError(f"No dev variants found for condition {condition!r}")
        score, _, variant = max(candidates)
        selected[condition] = variant
        selected_scores[condition] = score

    return {
        "version": 2,
        "task": metadata["task"],
        "model": metadata["model"],
        "seed": metadata["seed"],
        "dev_run": str(dev_run.resolve()),
        "primary_signal": signal,
        "selection_rule": "highest dev mean; lowest variant index breaks ties",
        "selected_variants": selected,
        "selected_dev_scores": selected_scores,
        "task_configuration": metadata.get("task_configuration"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock = select_variants(args.dev_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote prompt lock to {args.output}")


if __name__ == "__main__":
    main()
