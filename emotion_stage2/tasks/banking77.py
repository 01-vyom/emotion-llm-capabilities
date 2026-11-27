"""BANKING77 adapter for the generic Stage 2 runner."""

from __future__ import annotations

from collections import Counter
from typing import Any

from emotion_stage1.banking77 import (
    build_label_block,
    load_banking77_examples,
    load_fixture_examples,
    parse_prediction,
)

from ..reliability import is_refusal
from .base import Example, Parsed


class Banking77Task:
    name = "banking77"
    header = (
        "Classify the customer message into exactly one BANKING77 intent label. "
        "Return only the label string, with no explanation."
    )
    signal_keys = (
        "correct",
        "omission",
        "format_valid_wrong",
        "hallucination",
        "refusal",
        "parse_success",
    )
    primary_signals = ("correct",)

    def load(
        self,
        split: str,
        limit: int | None,
        seed: int,
    ) -> tuple[list[Example], dict[str, Any]]:
        dataset_split = "test" if split in {"dev", "test", "all"} else split
        source_examples, labels = load_banking77_examples(
            split=dataset_split,
            limit=None,
            seed=seed,
        )
        indices = list(range(len(source_examples)))
        if split in {"dev", "test"}:
            import random

            random.Random(seed).shuffle(indices)
            dev_size = max(1, round(0.2 * len(indices)))
            indices = indices[:dev_size] if split == "dev" else indices[dev_size:]
        if limit is not None:
            indices = indices[:limit]
        examples = [
            Example(
                item_id=source_examples[index].item_id,
                payload={"text": source_examples[index].text},
                gold=source_examples[index].gold_label,
            )
            for index in indices
        ]
        return examples, {"labels": labels}

    def load_fixture(self) -> tuple[list[Example], dict[str, Any]]:
        source_examples, labels = load_fixture_examples()
        examples = [
            Example(
                item_id=example.item_id,
                payload={"text": example.text},
                gold=example.gold_label,
            )
            for example in source_examples
        ]
        return examples, {"labels": labels}

    def build_item_block(
        self,
        example: Example,
        task_context: dict[str, Any],
    ) -> str:
        return (
            "[VALID LABELS]\n"
            f"{build_label_block(task_context['labels'])}\n\n"
            "[CUSTOMER MESSAGE]\n"
            f"{example.payload['text']}"
        )

    def parse(
        self,
        raw_output: str,
        example: Example,
        task_context: dict[str, Any],
    ) -> Parsed:
        del example
        pred, method = parse_prediction(raw_output, task_context["labels"])
        return Parsed(
            pred=pred,
            parse_method=method,
            raw_output=raw_output,
            details={"refusal": is_refusal(raw_output)},
        )

    def score(self, parsed: Parsed, example: Example) -> dict[str, float]:
        nonempty = bool(parsed.raw_output.strip())
        refusal = bool(parsed.details["refusal"])
        parsed_label = parsed.pred is not None
        correct = parsed.pred == example.gold
        return {
            "correct": float(correct),
            "omission": float(not parsed_label and not refusal),
            "format_valid_wrong": float(parsed_label and not correct),
            "hallucination": float(nonempty and not parsed_label and not refusal),
            "refusal": float(refusal),
            "parse_success": float(parsed_label),
        }

    def max_new_tokens(self) -> int:
        return 32

    def example_metadata(self, example: Example) -> dict[str, Any]:
        return {
            "text": example.payload["text"],
            "gold_label": example.gold,
        }

    def confusion_counts(self, rows: list[dict]) -> list[dict]:
        counter = Counter(
            (row["condition"], row["gold_label"], row["pred"] or "<unparsed>")
            for row in rows
            if not row["correct"]
        )
        return [
            {
                "condition": condition,
                "gold_label": gold,
                "pred_label": pred,
                "count": count,
            }
            for (condition, gold, pred), count in counter.most_common(100)
        ]

    def metadata(self) -> dict[str, Any]:
        return {"dataset": "PolyAI/banking77"}
