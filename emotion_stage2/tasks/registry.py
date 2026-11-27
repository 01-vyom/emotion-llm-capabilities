"""Task registry and CLI-driven construction."""

from __future__ import annotations

from argparse import Namespace

from .banking77 import Banking77Task
from .ifeval import IFEvalTask
from .ruler_niah import RulerNIAHTask


TASK_NAMES = ("banking77", "ifeval", "ruler_niah")


def make_task(name: str, args: Namespace):
    if name == "banking77":
        return Banking77Task()
    if name == "ifeval":
        return IFEvalTask()
    if name == "ruler_niah":
        hard = args.ruler_hard
        return RulerNIAHTask(
            model=args.model,
            context_lengths=tuple(args.ruler_context_lengths),
            depths=tuple(args.ruler_depths),
            modes=tuple(args.ruler_modes),
            samples_per_cell=args.ruler_samples_per_cell,
            hard=hard,
            distractor_count=(
                args.ruler_distractors
                if args.ruler_distractors is not None
                else (12 if hard else 0)
            ),
            multi_key_count=(
                args.ruler_multi_key_count
                if args.ruler_multi_key_count is not None
                else (6 if hard else 3)
            ),
            aggregate_match_count=(
                args.ruler_aggregate_match_count
                if args.ruler_aggregate_match_count is not None
                else (3 if hard else None)
            ),
            context_reserve_tokens=(
                args.ruler_context_reserve_tokens
                if args.ruler_context_reserve_tokens is not None
                else (1024 if hard else 0)
            ),
        )
    raise ValueError(f"Unknown task {name!r}")
