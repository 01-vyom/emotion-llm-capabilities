"""Stage 7 safety and IFEval arm definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import FULL_DOSE_GRID, PROMPT_CONDITIONS


@dataclass(frozen=True)
class Arm:
    name: str
    prompt_condition: str
    vector_state: str
    vector_kind: str | None
    alpha: float


def prompt_arms() -> list[Arm]:
    return [
        Arm(
            name=f"{condition}__none",
            prompt_condition=condition,
            vector_state="none",
            vector_kind=None,
            alpha=0.0,
        )
        for condition in PROMPT_CONDITIONS
    ]


def safety_arms(task_name: str, *, prompt_only: bool = False) -> list[Arm]:
    arms = prompt_arms()
    if task_name == "sycophancy" or prompt_only:
        return arms
    if task_name != "refusal_drift":
        raise ValueError(f"Unknown Stage 7 safety task {task_name!r}")
    for alpha in FULL_DOSE_GRID:
        for state, kind, sign in (
            ("positive", "unit", 1.0),
            ("negative", "unit", -1.0),
            ("random", "random_unit", 1.0),
            ("shuffled", "shuffled_unit", 1.0),
        ):
            arms.append(
                Arm(
                    name=f"neutral_length_matched__{state}_a{alpha:.1f}",
                    prompt_condition="neutral_length_matched",
                    vector_state=state,
                    vector_kind=kind,
                    alpha=sign * alpha,
                )
            )
    return arms


def ifeval_arms(*, prompt_only: bool = False) -> list[Arm]:
    arms = prompt_arms()
    if prompt_only:
        return arms
    for state, kind, sign in (
        ("positive", "unit", 1.0),
        ("negative", "unit", -1.0),
        ("random", "random_unit", 1.0),
        ("shuffled", "shuffled_unit", 1.0),
    ):
        arms.append(
            Arm(
                name=f"neutral_length_matched__{state}_a0.5",
                prompt_condition="neutral_length_matched",
                vector_state=state,
                vector_kind=kind,
                alpha=sign * 0.5,
            )
        )
    return arms

