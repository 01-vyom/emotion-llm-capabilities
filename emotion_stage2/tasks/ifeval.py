"""Official-verifier IFEval adapter."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from ..reliability import is_refusal
from .base import Example, Parsed


def _compact_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


class IFEvalTask:
    name = "ifeval"
    header = (
        "Respond to the user request below. Satisfy every explicit constraint "
        "in the request; do not discuss the constraints separately."
    )
    signal_keys = (
        "correct",
        "strict_prompt",
        "loose_prompt",
        "strict_instruction_rate",
        "loose_instruction_rate",
        "omission",
        "format_valid_wrong",
        "refusal",
        "parse_success",
    )
    primary_signals = ("strict_prompt",)

    def load(
        self,
        split: str,
        limit: int | None,
        seed: int,
    ) -> tuple[list[Example], dict[str, Any]]:
        from datasets import load_dataset

        dataset = load_dataset("google/IFEval", split="train")
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        if split in {"dev", "test"}:
            dev_size = max(1, round(0.2 * len(indices)))
            indices = indices[:dev_size] if split == "dev" else indices[dev_size:]
        elif split not in {"all", "train"}:
            raise ValueError(f"IFEval split must be dev, test, or all; got {split!r}")
        if limit is not None:
            indices = indices[:limit]

        examples = []
        for index in indices:
            row = dataset[index]
            examples.append(
                Example(
                    item_id=int(row["key"]),
                    payload={"prompt": str(row["prompt"])},
                    gold={
                        "instruction_id_list": list(row["instruction_id_list"]),
                        "kwargs": [
                            _compact_kwargs(dict(kwargs))
                            for kwargs in row["kwargs"]
                        ],
                    },
                )
            )
        return examples, {}

    def load_fixture(self) -> tuple[list[Example], dict[str, Any]]:
        examples = [
            Example(
                item_id=0,
                payload={
                    "prompt": (
                        "Write one short sentence containing the word cedar. "
                        "Do not use any commas."
                    )
                },
                gold={
                    "instruction_id_list": [
                        "keywords:existence",
                        "punctuation:no_comma",
                    ],
                    "kwargs": [
                        {"keywords": ["cedar"]},
                        {},
                    ],
                },
            )
        ]
        return examples, {}

    def build_item_block(
        self,
        example: Example,
        task_context: dict[str, Any],
    ) -> str:
        del task_context
        return f"[USER REQUEST]\n{example.payload['prompt']}"

    def parse(
        self,
        raw_output: str,
        example: Example,
        task_context: dict[str, Any],
    ) -> Parsed:
        del task_context
        from instruction_following_eval.evaluation_lib import (
            InputExample,
            test_instruction_following_loose,
            test_instruction_following_strict,
        )

        prompt = example.payload["prompt"]
        verifier_input = InputExample(
            key=example.item_id,
            instruction_id_list=example.gold["instruction_id_list"],
            prompt=prompt,
            kwargs=example.gold["kwargs"],
        )
        responses = {prompt: raw_output}
        instruction_ids = example.gold["instruction_id_list"]
        verifier_errors = {}
        try:
            strict = test_instruction_following_strict(verifier_input, responses)
            strict_results = strict.follow_instruction_list
            strict_prompt = strict.follow_all_instructions
        except Exception as exc:
            strict_results = [False] * len(instruction_ids)
            strict_prompt = False
            verifier_errors["strict"] = type(exc).__name__
        try:
            loose = test_instruction_following_loose(verifier_input, responses)
            loose_results = loose.follow_instruction_list
            loose_prompt = loose.follow_all_instructions
        except Exception as exc:
            loose_results = [False] * len(instruction_ids)
            loose_prompt = False
            verifier_errors["loose"] = type(exc).__name__
        failed_strict = [
            instruction_id
            for instruction_id, followed in zip(
                instruction_ids,
                strict_results,
                strict=True,
            )
            if not followed
        ]
        failed_loose = [
            instruction_id
            for instruction_id, followed in zip(
                instruction_ids,
                loose_results,
                strict=True,
            )
            if not followed
        ]
        return Parsed(
            pred=raw_output if raw_output.strip() else None,
            parse_method="nonempty" if raw_output.strip() else "empty",
            raw_output=raw_output,
            details={
                "strict_prompt": strict_prompt,
                "loose_prompt": loose_prompt,
                "strict_instruction_results": strict_results,
                "loose_instruction_results": loose_results,
                "failed_strict_instruction_ids": failed_strict,
                "failed_loose_instruction_ids": failed_loose,
                "verifier_errors": verifier_errors,
                "refusal": is_refusal(raw_output),
            },
        )

    def score(self, parsed: Parsed, example: Example) -> dict[str, float]:
        del example
        strict_results = parsed.details["strict_instruction_results"]
        loose_results = parsed.details["loose_instruction_results"]
        strict_prompt = bool(parsed.details["strict_prompt"])
        loose_prompt = bool(parsed.details["loose_prompt"])
        refusal = bool(parsed.details["refusal"])
        nonempty = bool(parsed.raw_output.strip())
        return {
            "correct": float(strict_prompt),
            "strict_prompt": float(strict_prompt),
            "loose_prompt": float(loose_prompt),
            "strict_instruction_rate": (
                sum(strict_results) / len(strict_results) if strict_results else 0.0
            ),
            "loose_instruction_rate": (
                sum(loose_results) / len(loose_results) if loose_results else 0.0
            ),
            "omission": float(nonempty and not refusal and not strict_prompt),
            "format_valid_wrong": float(loose_prompt and not strict_prompt),
            "refusal": float(refusal),
            "parse_success": float(nonempty),
        }

    def max_new_tokens(self) -> int:
        return 4096

    def example_metadata(self, example: Example) -> dict[str, Any]:
        return {
            "source_prompt": example.payload["prompt"],
            "instruction_id_list": example.gold["instruction_id_list"],
            "instruction_kwargs": example.gold["kwargs"],
        }

    def confusion_counts(self, rows: list[dict]) -> list[dict]:
        counter = Counter()
        for row in rows:
            for instruction_id in row["details"]["failed_strict_instruction_ids"]:
                counter[(row["condition"], instruction_id)] += 1
        return [
            {
                "condition": condition,
                "failed_instruction_id": instruction_id,
                "count": count,
            }
            for (condition, instruction_id), count in counter.most_common()
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": "google/IFEval",
            "dataset_source_split": "train",
            "dev_fraction": 0.2,
            "verifier": "vendored google-research/instruction_following_eval",
        }
