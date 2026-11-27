"""Freeze Stage 5 DV admissions before any crossed-panel treatment run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .admission import ADMISSION_CONDITION
from .tasks import TASK_NAMES


def make_panel_lock(admission_dirs: list[Path]) -> dict:
    by_task = {}
    models = set()
    for directory in admission_dirs:
        admission = json.loads(
            (directory / "admission.json").read_text(encoding="utf-8")
        )
        task = admission["task"]
        if task in by_task:
            raise ValueError(f"Duplicate admission artifact for {task}")
        if admission.get("stage") != 5 or admission.get("version") != 1:
            raise ValueError(f"Invalid Stage 5 admission artifact: {directory}")
        if not admission.get("lockable"):
            raise ValueError(f"Admission artifact is not lockable: {directory}")
        if admission.get("condition") != ADMISSION_CONDITION:
            raise ValueError(f"Unexpected admission condition: {directory}")
        by_task[task] = {
            **admission,
            "admission_dir": str(directory.resolve()),
        }
        models.add(admission["model"])
    if set(by_task) != set(TASK_NAMES):
        missing = sorted(set(TASK_NAMES) - set(by_task))
        extra = sorted(set(by_task) - set(TASK_NAMES))
        raise ValueError(f"Admission task mismatch; missing={missing}, extra={extra}")
    if len(models) != 1:
        raise ValueError(f"Admission artifacts use multiple models: {models}")

    admitted_tasks = [
        task for task in TASK_NAMES if by_task[task]["admitted_signals"]
    ]
    return {
        "version": 1,
        "stage": 5,
        "artifact": "panel_lock",
        "model": next(iter(models)),
        "admission_condition": ADMISSION_CONDITION,
        "tasks": by_task,
        "admitted_tasks": admitted_tasks,
        "admitted_signals": [
            {"task": task, "signal": signal}
            for task in TASK_NAMES
            for signal in by_task[task]["admitted_signals"]
        ],
        "not_testable_signals": [
            {"task": task, "signal": signal}
            for task in TASK_NAMES
            for signal in by_task[task]["not_testable_signals"]
        ],
        "selection_rule": (
            "DV-level admission on the complete held-out test pool; no individual "
            "test item is selected or removed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--admission-dirs",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    lock = make_panel_lock(args.admission_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
