"""Deterministic RULER-style needle-in-a-haystack adapter."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from ..reliability import is_refusal
from .base import Example, Parsed


_CODE_RE = re.compile(
    r"[A-Z]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}",
    flags=re.IGNORECASE,
)
_COUNT_RE = re.compile(r"(?<![\w-])\d+(?![\w-])")
_FILLER_TEMPLATES = (
    "Review entry {index:06d}: team {team} checked account {reference}; "
    "status {status}; follow-up {follow_up}.",
    "Ledger item {index:06d}: unit {team} reviewed file {reference}; "
    "result {status}; action {follow_up}.",
    "Daily note {index:06d}: group {team} inspected record {reference}; "
    "outcome {status}; next step {follow_up}.",
    "Control log {index:06d}: desk {team} examined request {reference}; "
    "state {status}; routing {follow_up}.",
)
_TEAMS = ("amber", "birch", "cedar", "delta", "elm", "fjord")
_STATUSES = ("cleared", "routine", "closed", "stable", "verified")
_FOLLOW_UPS = ("none", "standard", "scheduled", "archived")


class RulerNIAHTask:
    name = "ruler_niah"
    header = (
        "Read the routine document and retrieve the requested rare value or "
        "values. Use only the document and follow the query's exact output format."
    )
    signal_keys = (
        "correct",
        "retrieval_rate",
        "omission",
        "format_valid_wrong",
        "hallucination",
        "refusal",
        "parse_success",
    )
    primary_signals = ("correct",)

    def __init__(
        self,
        model: str,
        context_lengths: tuple[int, ...] = (4096, 8192, 16384),
        depths: tuple[float, ...] = (0.1, 0.5, 0.9),
        modes: tuple[str, ...] = ("single", "multi"),
        samples_per_cell: int = 4,
        hard: bool = False,
        distractor_count: int = 0,
        multi_key_count: int = 3,
        aggregate_match_count: int | None = None,
        context_reserve_tokens: int = 0,
    ):
        unknown_modes = sorted(set(modes) - {"single", "multi", "aggregate"})
        if unknown_modes:
            raise ValueError(f"Unknown RULER modes: {unknown_modes}")
        if "aggregate" in modes and not hard:
            raise ValueError("RULER aggregate mode requires hard=True")
        if any(context_length <= 0 for context_length in context_lengths):
            raise ValueError("RULER context lengths must be positive")
        if any(not 0.0 <= depth <= 1.0 for depth in depths):
            raise ValueError("RULER depths must be between 0 and 1")
        if samples_per_cell <= 0:
            raise ValueError("RULER samples per cell must be positive")
        if distractor_count < 0:
            raise ValueError("RULER distractor count cannot be negative")
        if multi_key_count <= 0:
            raise ValueError("RULER multi-key count must be positive")
        if aggregate_match_count is not None and aggregate_match_count <= 0:
            raise ValueError("RULER aggregate match count must be positive")
        if context_reserve_tokens < 0:
            raise ValueError("RULER context reserve cannot be negative")
        self.model = model
        self.context_lengths = context_lengths
        self.depths = depths
        self.modes = modes
        self.samples_per_cell = samples_per_cell
        self.hard = hard
        self.distractor_count = distractor_count
        self.multi_key_count = multi_key_count
        self.aggregate_match_count = (
            multi_key_count
            if aggregate_match_count is None
            else aggregate_match_count
        )
        self.context_reserve_tokens = context_reserve_tokens
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                trust_remote_code=True,
            )
        return self._tokenizer

    @staticmethod
    def _code(seed_text: str) -> str:
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest().upper()
        return f"NIAH-{digest[:4]}-{digest[4:8]}"

    @staticmethod
    def _case_name(seed_text: str, index: int) -> str:
        digest = hashlib.sha256(f"{seed_text}:{index}".encode("utf-8")).hexdigest()
        return f"case-{digest[:10]}"

    @staticmethod
    def _record_number(seed_text: str, record_type: str, index: int) -> int:
        digest = hashlib.sha256(
            f"{seed_text}:{record_type}:{index}".encode("utf-8")
        ).hexdigest()
        return int(digest[:8], 16) % 1_000_000

    def _hard_filler_tokens(
        self,
        seed_text: str,
        token_budget: int,
    ) -> list[int]:
        tokenizer = self._get_tokenizer()
        base = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
        record_count = max(64, token_budget // 18)
        while True:
            lines = []
            for index in range(record_count):
                template = _FILLER_TEMPLATES[(base + index * 7) % len(_FILLER_TEMPLATES)]
                lines.append(
                    template.format(
                        index=index,
                        team=_TEAMS[(base + index * 5) % len(_TEAMS)],
                        reference=f"ref-{(base + index * 7919) % 1_000_000:06d}",
                        status=_STATUSES[(base + index * 3) % len(_STATUSES)],
                        follow_up=_FOLLOW_UPS[
                            (base + index * 11) % len(_FOLLOW_UPS)
                        ],
                    )
                )
            tokens = tokenizer.encode("\n".join(lines), add_special_tokens=False)
            if len(tokens) >= token_budget:
                return tokens[:token_budget]
            record_count = int(record_count * 1.25) + 1

    @staticmethod
    def _decoy_depths(count: int, target_depth: float) -> list[float]:
        depths = []
        for index in range(count):
            depth = (index + 1) / (count + 1)
            if abs(depth - target_depth) < 0.025:
                depth = min(0.99, depth + 0.04)
            depths.append(depth)
        return depths

    @staticmethod
    def _insert_blocks(
        filler_tokens: list[int],
        blocks: list[tuple[float, list[int]]],
    ) -> list[int]:
        pieces: list[int] = []
        cursor = 0
        for depth, block_tokens in sorted(blocks, key=lambda item: item[0]):
            insertion = round(depth * len(filler_tokens))
            pieces.extend(filler_tokens[cursor:insertion])
            pieces.extend(block_tokens)
            cursor = insertion
        pieces.extend(filler_tokens[cursor:])
        return pieces

    def _build_hard_example(
        self,
        item_id: int,
        split: str,
        context_length: int,
        depth: float,
        mode: str,
        sample: int,
        seed: int,
    ) -> Example:
        tokenizer = self._get_tokenizer()
        seed_text = (
            f"hard:{split}:{context_length}:{mode}:{sample}:{seed}"
        )
        desired_context_tokens = context_length - self.context_reserve_tokens
        if desired_context_tokens <= 0:
            raise ValueError(
                "RULER context reserve must be smaller than every context length"
            )

        cases: list[str] = []
        aggregate_marker: str | None = None
        aggregate_count: int | None = None
        if mode == "aggregate":
            aggregate_count = self.aggregate_match_count
            digest = hashlib.sha256(f"{seed_text}:marker".encode("utf-8")).hexdigest()
            aggregate_marker = f"tag-{digest[:10]}"
            target_lines = [
                (
                    "Review entry "
                    f"{self._record_number(seed_text, 'target', index):06d}: "
                    "supporting note mentions "
                    f"{aggregate_marker}; status routine; follow-up standard."
                )
                for index in range(aggregate_count)
            ]
            gold: Any = aggregate_count
            query = (
                f"How many review entries mention {aggregate_marker}? "
                "Return only the integer count."
            )
        else:
            target_count = 1 if mode == "single" else self.multi_key_count
            cases = [
                self._case_name(seed_text, index) for index in range(target_count)
            ]
            targets = [
                self._code(f"{seed_text}:{case}") for case in cases
            ]
            target_lines = [
                (
                    "Review entry "
                    f"{self._record_number(seed_text, 'target', index):06d}: "
                    f"account {case} has access "
                    f"code {target}; status routine; follow-up standard."
                )
                for index, (case, target) in enumerate(
                    zip(cases, targets, strict=True)
                )
            ]
            gold = targets
            if mode == "single":
                query = (
                    f"What is the access code for {cases[0]}? "
                    "Return only the exact code."
                )
            else:
                case_list = ", ".join(cases)
                query = (
                    f"Give the access codes for {case_list}, in that order. "
                    'Return only the codes separated by " | ".'
                )

        decoy_lines = []
        for index in range(self.distractor_count):
            if mode == "aggregate":
                digest = hashlib.sha256(
                    f"{seed_text}:decoy-marker:{index}".encode("utf-8")
                ).hexdigest()
                decoy_marker = f"tag-{digest[:10]}"
                decoy_lines.append(
                    "Review entry "
                    f"{self._record_number(seed_text, 'decoy', index):06d}: "
                    "supporting note mentions "
                    f"{decoy_marker}; status routine; follow-up standard."
                )
            else:
                decoy_case = self._case_name(f"{seed_text}:decoy", index)
                decoy_code = self._code(f"{seed_text}:decoy:{decoy_case}")
                decoy_lines.append(
                    "Review entry "
                    f"{self._record_number(seed_text, 'decoy', index):06d}: "
                    f"account {decoy_case} has "
                    f"access code {decoy_code}; status routine; follow-up standard."
                )

        target_tokens = tokenizer.encode(
            "\n" + "\n".join(target_lines) + "\n",
            add_special_tokens=False,
        )
        decoy_token_blocks = [
            tokenizer.encode(f"\n{line}\n", add_special_tokens=False)
            for line in decoy_lines
        ]
        inserted_token_count = len(target_tokens) + sum(
            len(tokens) for tokens in decoy_token_blocks
        )
        filler_budget = desired_context_tokens - inserted_token_count
        if filler_budget <= 0:
            raise ValueError(
                "RULER inserts exceed the context budget; reduce distractors or "
                "increase the context length"
            )
        filler_tokens = self._hard_filler_tokens(seed_text, filler_budget)
        blocks = [(depth, target_tokens)]
        blocks.extend(
            zip(
                self._decoy_depths(self.distractor_count, depth),
                decoy_token_blocks,
                strict=True,
            )
        )
        context_tokens = self._insert_blocks(filler_tokens, blocks)
        context = tokenizer.decode(context_tokens, skip_special_tokens=True)
        actual_tokens = len(tokenizer.encode(context, add_special_tokens=False))

        return Example(
            item_id=item_id,
            payload={
                "context": context,
                "query": query,
                "context_length_target": context_length,
                "context_length_actual": actual_tokens,
                "context_reserve_tokens": self.context_reserve_tokens,
                "needle_depth": depth,
                "mode": mode,
                "cases": cases,
                "aggregate_marker": aggregate_marker,
                "aggregate_count": aggregate_count,
                "distractor_count": self.distractor_count,
            },
            gold=gold,
        )

    def _build_example(
        self,
        item_id: int,
        split: str,
        context_length: int,
        depth: float,
        mode: str,
        sample: int,
        seed: int,
    ) -> Example:
        if self.hard:
            return self._build_hard_example(
                item_id=item_id,
                split=split,
                context_length=context_length,
                depth=depth,
                mode=mode,
                sample=sample,
                seed=seed,
            )
        tokenizer = self._get_tokenizer()
        seed_text = (
            f"{split}:{context_length}:{depth:.3f}:{mode}:{sample}:{seed}"
        )
        target_count = 1 if mode == "single" else 3
        cases = [
            self._case_name(seed_text, index) for index in range(target_count)
        ]
        targets = [
            self._code(f"{seed_text}:{case}") for case in cases
        ]
        needle_lines = [
            (
                "IMPORTANT EXCEPTION RECORD: "
                f"The access code for {case} is {target}."
            )
            for case, target in zip(cases, targets, strict=True)
        ]
        needle = "\n" + "\n".join(needle_lines) + "\n"
        needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
        filler_budget = max(1, context_length - len(needle_tokens))

        record_count = max(32, filler_budget // 10)
        while True:
            filler = "\n".join(
                (
                    f"Routine record {index:06d}: review complete; status normal; "
                    "no exception noted."
                )
                for index in range(record_count)
            )
            filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
            if len(filler_tokens) >= filler_budget:
                break
            record_count = int(record_count * 1.25) + 1
        filler_tokens = filler_tokens[:filler_budget]
        insertion = round(depth * len(filler_tokens))
        context_tokens = (
            filler_tokens[:insertion]
            + needle_tokens
            + filler_tokens[insertion:]
        )
        context = tokenizer.decode(context_tokens, skip_special_tokens=True)
        actual_tokens = len(tokenizer.encode(context, add_special_tokens=False))

        if mode == "single":
            query = (
                f"What is the access code for {cases[0]}? "
                "Return only the exact code."
            )
        else:
            case_list = ", ".join(cases)
            query = (
                f"Give the access codes for {case_list}, in that order. "
                'Return only the codes separated by " | ".'
            )
        return Example(
            item_id=item_id,
            payload={
                "context": context,
                "query": query,
                "context_length_target": context_length,
                "context_length_actual": actual_tokens,
                "needle_depth": depth,
                "mode": mode,
                "cases": cases,
            },
            gold=targets,
        )

    def load(
        self,
        split: str,
        limit: int | None,
        seed: int,
    ) -> tuple[list[Example], dict[str, Any]]:
        if split not in {"dev", "test", "all"}:
            raise ValueError(f"RULER split must be dev, test, or all; got {split!r}")
        samples = 1 if split == "dev" else self.samples_per_cell
        if split == "all":
            samples += 1
        examples = []
        item_id = 0
        for context_length in self.context_lengths:
            for depth in self.depths:
                for mode in self.modes:
                    for sample in range(samples):
                        examples.append(
                            self._build_example(
                                item_id=item_id,
                                split=split,
                                context_length=context_length,
                                depth=depth,
                                mode=mode,
                                sample=sample,
                                seed=seed,
                            )
                        )
                        item_id += 1
        if limit is not None:
            examples = examples[:limit]
        return examples, {}

    def load_fixture(self) -> tuple[list[Example], dict[str, Any]]:
        target = "NIAH-AB12-CD34"
        return [
            Example(
                item_id=0,
                payload={
                    "context": (
                        "Routine record: status normal.\n"
                        "IMPORTANT EXCEPTION RECORD: The access code for "
                        f"case-fixture is {target}.\n"
                        "Routine record: review complete."
                    ),
                    "query": (
                        "What is the access code for case-fixture? "
                        "Return only the exact code."
                    ),
                    "context_length_target": 64,
                    "context_length_actual": 47,
                    "needle_depth": 0.5,
                    "mode": "single",
                    "cases": ["case-fixture"],
                },
                gold=[target],
            )
        ], {}

    def build_item_block(
        self,
        example: Example,
        task_context: dict[str, Any],
    ) -> str:
        del task_context
        return (
            "[ROUTINE DOCUMENT]\n"
            f"{example.payload['context']}\n\n"
            "[QUERY]\n"
            f"{example.payload['query']}"
        )

    def parse(
        self,
        raw_output: str,
        example: Example,
        task_context: dict[str, Any],
    ) -> Parsed:
        del task_context
        if example.payload.get("mode") == "aggregate":
            found_counts = [int(match) for match in _COUNT_RE.findall(raw_output)]
            return Parsed(
                pred=found_counts or None,
                parse_method="counts_found" if found_counts else "no_count",
                raw_output=raw_output,
                details={
                    "found_counts": found_counts,
                    "found_codes": [],
                    "refusal": is_refusal(raw_output),
                },
            )
        found = [match.upper() for match in _CODE_RE.findall(raw_output)]
        return Parsed(
            pred=found or None,
            parse_method="codes_found" if found else "no_code",
            raw_output=raw_output,
            details={
                "found_codes": found,
                "found_counts": [],
                "refusal": is_refusal(raw_output),
            },
        )

    def score(self, parsed: Parsed, example: Example) -> dict[str, float]:
        if example.payload.get("mode") == "aggregate":
            expected_count = int(example.gold)
            found_counts = parsed.details["found_counts"]
            refusal = bool(parsed.details["refusal"])
            normalized_output = parsed.raw_output.strip()
            format_valid = bool(re.fullmatch(r"\d+", normalized_output))
            predicted_count = (
                int(normalized_output) if format_valid else None
            )
            correct = format_valid and predicted_count == expected_count
            return {
                "correct": float(correct),
                "retrieval_rate": float(correct),
                "omission": float(not found_counts and not refusal),
                "format_valid_wrong": float(format_valid and not correct),
                "hallucination": float(
                    predicted_count is not None and predicted_count != expected_count
                ),
                "refusal": float(refusal),
                "parse_success": float(bool(found_counts)),
            }

        expected = [value.upper() for value in example.gold]
        found = parsed.details["found_codes"]
        expected_set = set(expected)
        found_set = set(found)
        retrieved = len(expected_set & found_set)
        refusal = bool(parsed.details["refusal"])
        normalized_output = parsed.raw_output.strip()
        format_valid = (
            len(found) == len(expected)
            and normalized_output == " | ".join(found)
        )
        correct = format_valid and found == expected
        return {
            "correct": float(correct),
            "retrieval_rate": retrieved / len(expected_set),
            "omission": float(bool(expected_set - found_set) and not refusal),
            "format_valid_wrong": float(format_valid and not correct),
            "hallucination": float(bool(found_set - expected_set)),
            "refusal": float(refusal),
            "parse_success": float(bool(found)),
        }

    def max_new_tokens(self) -> int:
        return 128 if self.hard else 64

    def example_metadata(self, example: Example) -> dict[str, Any]:
        return {
            "query": example.payload["query"],
            "gold": example.gold,
            "context_length_target": example.payload["context_length_target"],
            "context_length_actual": example.payload["context_length_actual"],
            "needle_depth": example.payload["needle_depth"],
            "mode": example.payload["mode"],
            "cases": example.payload["cases"],
            "context_reserve_tokens": example.payload.get(
                "context_reserve_tokens",
                0,
            ),
            "aggregate_marker": example.payload.get("aggregate_marker"),
            "aggregate_count": example.payload.get("aggregate_count"),
            "distractor_count": example.payload.get("distractor_count", 0),
        }

    def confusion_counts(self, rows: list[dict]) -> list[dict]:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            key = (
                row["condition"],
                row["mode"],
                row["context_length_target"],
                row["needle_depth"],
            )
            groups[key].append(row)
        output = []
        for key, group in sorted(groups.items()):
            condition, mode, context_length, depth = key
            output.append(
                {
                    "condition": condition,
                    "mode": mode,
                    "context_length": context_length,
                    "needle_depth": depth,
                    "n": len(group),
                    "accuracy": sum(row["correct"] for row in group) / len(group),
                    "retrieval_rate": (
                        sum(row["retrieval_rate"] for row in group) / len(group)
                    ),
                    "omission_rate": (
                        sum(row["omission"] for row in group) / len(group)
                    ),
                }
            )
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": "synthetic RULER-style NIAH",
            "context_lengths": list(self.context_lengths),
            "needle_depths": list(self.depths),
            "modes": list(self.modes),
            "test_samples_per_cell": self.samples_per_cell,
            "hard": self.hard,
            "distractor_count": self.distractor_count,
            "multi_key_count": self.multi_key_count,
            "aggregate_match_count": self.aggregate_match_count,
            "context_reserve_tokens": self.context_reserve_tokens,
        }
