"""BANKING77 loading, prompt construction, and parsing."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable

from .affect import build_affect_block


DATASET_NAME = "PolyAI/banking77"


@dataclass(frozen=True)
class BankingExample:
    item_id: int
    text: str
    gold_label: str


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("`", "").replace('"', "").replace("'", "")
    text = re.sub(r"^intent\s*:\s*", "", text)
    text = re.sub(r"^label\s*:\s*", "", text)
    text = text.strip(" .,\n\t")
    text = re.sub(r"[\s-]+", "_", text)
    return text


def load_banking77_examples(
    split: str = "test",
    limit: int | None = None,
    seed: int = 13,
) -> tuple[list[BankingExample], list[str]]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split=split)
    labels = dataset.features["label"].names

    indices = list(range(len(dataset)))
    if limit is not None and limit < len(indices):
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:limit]

    examples = [
        BankingExample(
            item_id=int(i),
            text=str(dataset[i]["text"]),
            gold_label=labels[int(dataset[i]["label"])],
        )
        for i in indices
    ]
    return examples, list(labels)


def load_fixture_examples() -> tuple[list[BankingExample], list[str]]:
    """Small dependency-free fixture for smoke tests."""

    labels = [
        "activate_my_card",
        "card_arrival",
        "cash_withdrawal_charge",
        "lost_or_stolen_card",
        "transfer_timing",
        "verify_my_identity",
    ]
    examples = [
        BankingExample(0, "My new card still hasn't arrived. When should I expect it?", "card_arrival"),
        BankingExample(1, "How long does a transfer usually take to reach another bank?", "transfer_timing"),
        BankingExample(2, "I lost my card yesterday and need to block it now.", "lost_or_stolen_card"),
    ]
    return examples, labels


def build_label_block(labels: Iterable[str]) -> str:
    return "\n".join(f"- {label}" for label in labels)


def build_prompt(example: BankingExample, labels: list[str], condition_name: str) -> str:
    """Construct one classification prompt.

    The label set and instructions are identical across affect conditions; only
    the affect cue changes. The answer format is intentionally strict to allow
    objective parsing.
    """

    return (
        "[TASK]\n"
        "Classify the customer message into exactly one BANKING77 intent label.\n"
        "Return only the label string, with no explanation.\n\n"
        f"{build_affect_block(condition_name)}\n\n"
        "[VALID LABELS]\n"
        f"{build_label_block(labels)}\n\n"
        "[CUSTOMER MESSAGE]\n"
        f"{example.text}\n\n"
        "[ANSWER]\n"
    )


def parse_prediction(raw_output: str, labels: list[str]) -> tuple[str | None, str]:
    """Parse a generated label from raw model output.

    Returns `(pred_label, parse_method)`. `pred_label` is `None` if no label can
    be recovered.
    """

    normalized_to_label = {normalize_label(label): label for label in labels}
    first_line = raw_output.strip().splitlines()[0] if raw_output.strip() else ""
    normalized_first = normalize_label(first_line)
    if normalized_first in normalized_to_label:
        return normalized_to_label[normalized_first], "first_line_exact"

    normalized_raw = normalize_label(raw_output)
    if normalized_raw in normalized_to_label:
        return normalized_to_label[normalized_raw], "full_output_exact"

    compact_raw = normalize_label(raw_output)
    matches = []
    for norm, label in normalized_to_label.items():
        if norm in compact_raw:
            matches.append(label)
    if len(matches) == 1:
        return matches[0], "single_label_mentioned"

    return None, "unparsed"
