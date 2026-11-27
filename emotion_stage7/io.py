"""Stage 7 atomic checkpoint and lock helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from emotion_stage3.io import read_jsonl, sha256_file


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def checkpoint_rows(path: Path, item_ids: list[int]) -> list[dict] | None:
    if not path.exists():
        return None
    rows = read_jsonl(path)
    if [int(row["item_id"]) for row in rows] != item_ids:
        return None
    return rows if all("raw_output" in row for row in rows) else None


def load_panel_lock(path: Path, *, model: str) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("stage") != 7
        or lock.get("version") != 1
        or lock.get("artifact") != "panel_lock"
    ):
        raise ValueError("Invalid Stage 7 panel lock")
    if lock.get("model") != model:
        raise ValueError("Stage 7 panel lock model does not match")
    return lock


def load_steering_lock(
    path: Path,
    *,
    vector_dir: Path,
    model: str,
) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("model") != model:
        raise ValueError("Steering lock model does not match")
    if lock.get("direction") != "diligence" or lock.get("policy") != "last":
        raise ValueError("Stage 7 requires diligence last-token steering")
    if lock.get("vectors_sha256") != sha256_file(vector_dir / "vectors.npz"):
        raise ValueError("Steering lock vector hash does not match")
    return lock


def seed_baseline_checkpoint(
    *,
    admission_dir: Path,
    checkpoint: Path,
    item_ids: list[int],
    bedrock: bool,
) -> None:
    rows = read_jsonl(admission_dir / "predictions.jsonl")
    by_item = {int(row["item_id"]): row for row in rows}
    if set(by_item) != set(item_ids):
        raise ValueError("Admission and treatment item IDs differ")
    fields = (
        (
            "stop_reason",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "latency_ms",
        )
        if bedrock
        else ()
    )
    write_jsonl_atomic(
        checkpoint,
        [
            {
                "item_id": item_id,
                "raw_output": by_item[item_id]["raw_output"],
                **{field: by_item[item_id][field] for field in fields},
                "reused_from_admission": True,
            }
            for item_id in item_ids
        ],
    )

