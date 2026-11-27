"""Shared task interface and prompt assembly for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from emotion_stage1.affect import build_affect_block


@dataclass(frozen=True)
class Example:
    item_id: int
    payload: dict[str, Any]
    gold: Any


@dataclass(frozen=True)
class Parsed:
    pred: Any | None
    parse_method: str
    raw_output: str
    details: dict[str, Any] = field(default_factory=dict)


class Task(Protocol):
    name: str
    header: str
    signal_keys: tuple[str, ...]
    primary_signals: tuple[str, ...]

    def load(
        self,
        split: str,
        limit: int | None,
        seed: int,
    ) -> tuple[list[Example], dict[str, Any]]:
        ...

    def load_fixture(self) -> tuple[list[Example], dict[str, Any]]:
        ...

    def build_item_block(
        self,
        example: Example,
        task_context: dict[str, Any],
    ) -> str:
        ...

    def parse(
        self,
        raw_output: str,
        example: Example,
        task_context: dict[str, Any],
    ) -> Parsed:
        ...

    def score(self, parsed: Parsed, example: Example) -> dict[str, float]:
        ...

    def max_new_tokens(self) -> int:
        ...

    def example_metadata(self, example: Example) -> dict[str, Any]:
        ...

    def confusion_counts(self, rows: list[dict]) -> list[dict]:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


def build_prompt(
    task: Task,
    example: Example,
    task_context: dict[str, Any],
    condition: str,
    variant_index: int = 0,
) -> str:
    return (
        "[TASK]\n"
        f"{task.header}\n\n"
        f"{build_affect_block(condition, variant_index=variant_index)}\n\n"
        f"{task.build_item_block(example, task_context)}\n\n"
        "[ANSWER]\n"
    )


def variant_arm(condition: str, variant_index: int, label_variant: bool) -> str:
    if label_variant:
        return f"{condition}::v{variant_index}"
    return condition
