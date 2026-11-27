"""Freeze one model's Stage 7 difficulty and held-out admission results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import PREREGISTRATION_SHA256, TASK_NAMES


def make_panel_lock(admission_root: Path) -> dict:
    tasks = {}
    models = set()
    engines = set()
    for task_name in TASK_NAMES:
        directory = admission_root / task_name
        admission = json.loads(
            (directory / "admission.json").read_text(encoding="utf-8")
        )
        if (
            admission.get("stage") != 7
            or admission.get("artifact") != "held_out_dv_admission"
            or admission.get("task") != task_name
            or not admission.get("lockable")
        ):
            raise ValueError(f"Invalid Stage 7 admission artifact: {directory}")
        if admission["preregistration_sha256"] != PREREGISTRATION_SHA256:
            raise ValueError("Admission preregistration hash does not match")
        tasks[task_name] = {
            **admission,
            "admission_dir": str(directory.resolve()),
        }
        models.add(admission["model"])
        engines.add(admission["engine"])
    if len(models) != 1 or len(engines) != 1:
        raise ValueError("Stage 7 admissions do not share one model and engine")
    return {
        "version": 1,
        "stage": 7,
        "artifact": "panel_lock",
        "model": next(iter(models)),
        "engine": next(iter(engines)),
        "tasks": tasks,
        "admitted_tasks": [
            task_name
            for task_name in TASK_NAMES
            if tasks[task_name]["admitted_signals"]
        ],
        "admitted_signals": [
            {"task": task_name, "signal": signal}
            for task_name in TASK_NAMES
            for signal in tasks[task_name]["admitted_signals"]
        ],
        "not_testable_signals": [
            {"task": task_name, "signal": signal}
            for task_name in TASK_NAMES
            for signal in tasks[task_name]["not_testable_signals"]
        ],
        "selection_rule": (
            "development-only tier-mixture lock followed by independent held-out "
            "base-rate admission; no held-out item reselection"
        ),
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    lock = make_panel_lock(args.admission_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
