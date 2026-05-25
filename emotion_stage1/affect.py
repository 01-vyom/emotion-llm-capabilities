"""Affective prompt conditions and Anthropic-paper-inspired emotion stimuli.

The linked Anthropic paper did not identify emotion concepts by simply asking a
model to "be emotional." It generated short synthetic stories where a character
experienced a target emotion, collected residual-stream activations, averaged
within each emotion, and used those directions as emotion probes/vectors.

Stage 1 is prompt-only, so the conditions below are behavioral affect cues. The
emotion concepts and story prompt builder are included to keep Stage 1 aligned
with the later activation-vector study.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


BASE_INSTRUCTION = (
    "Follow the task instructions exactly. Use only the provided input. "
    "Do not invent missing information. Return only the requested output format."
)


@dataclass(frozen=True)
class AffectCondition:
    name: str
    category: str
    cue: str


CONDITIONS: dict[str, AffectCondition] = {
    "neutral_empty": AffectCondition(
        name="neutral_empty",
        category="neutral",
        cue="",
    ),
    "neutral_length_matched": AffectCondition(
        name="neutral_length_matched",
        category="neutral",
        cue=(
            "Continue with the same level of care that would normally be "
            "appropriate for this task. Keep the requested format and rely "
            "only on the provided input."
        ),
    ),
    "neutral_procedural": AffectCondition(
        name="neutral_procedural",
        category="neutral",
        cue=(
            "Process the input according to the instructions. Maintain "
            "consistency across all items and provide the final answer in the "
            "requested format."
        ),
    ),
    "positive_pride": AffectCondition(
        name="positive_pride",
        category="positive",
        cue=(
            "Take pride in doing this task well. Treat the details as worth "
            "getting right from start to finish."
        ),
    ),
    "negative_failure_avoidance": AffectCondition(
        name="negative_failure_avoidance",
        category="negative",
        cue=(
            "The main risk is missing details that should have been handled. "
            "Avoid that failure mode and keep the answer grounded in the input."
        ),
    ),
    "high_stakes_audit": AffectCondition(
        name="high_stakes_audit",
        category="high_stakes",
        cue=(
            "Treat this as an audit-quality task. Omissions, invented details, "
            "and format errors should be avoided."
        ),
    ),
    "high_stakes_extreme": AffectCondition(
        name="high_stakes_extreme",
        category="high_stakes",
        cue=(
            "Failure on this task would have serious consequences. Be precise, "
            "complete, and do not guess beyond the provided input."
        ),
    ),
    "calm_meticulous": AffectCondition(
        name="calm_meticulous",
        category="calm_focused",
        cue=(
            "Use a patient, meticulous working style. Do not rush, overreact, "
            "or add anything outside the instructions."
        ),
    ),
    "boredom_antidote_repetitive": AffectCondition(
        name="boredom_antidote_repetitive",
        category="boredom_antidote",
        cue=(
            "This task may be repetitive, but every item still matters. "
            "Maintain the same level of attention throughout."
        ),
    ),
}


DEFAULT_CONDITIONS = (
    "neutral_empty",
    "neutral_length_matched",
    "positive_pride",
    "negative_failure_avoidance",
    "high_stakes_audit",
    "high_stakes_extreme",
    "calm_meticulous",
    "boredom_antidote_repetitive",
)


# Small, task-relevant subset of the Anthropic paper's broader emotion concept
# space. These are enough for Stage 1 conditions and Stage 3 vector extraction.
EMOTION_VECTOR_CONCEPTS = (
    "calm",
    "desperate",
    "focused",
    "distracted",
    "patient",
    "rushed",
    "responsible",
    "careless",
    "curious",
    "bored",
)


EMOTION_STORY_TOPICS = (
    "An employee is asked to review a long table of nearly identical records",
    "A clerk must classify hundreds of similar customer-service messages",
    "An auditor checks a routine report for one rare but important error",
    "A programmer repairs a small repeated bug across many files",
    "A legal assistant reviews clauses that mostly look irrelevant",
    "A data analyst normalizes dates and amounts in a messy spreadsheet",
)


def get_condition(name: str) -> AffectCondition:
    try:
        return CONDITIONS[name]
    except KeyError as exc:
        known = ", ".join(sorted(CONDITIONS))
        raise ValueError(f"Unknown affect condition {name!r}. Known: {known}") from exc


def build_affect_block(condition_name: str) -> str:
    condition = get_condition(condition_name)
    if condition.cue:
        return f"{BASE_INSTRUCTION}\n\n[AFFECT CUE]\n{condition.cue}"
    return BASE_INSTRUCTION


def build_emotion_story_prompt(emotion: str, topic: str, replicate: int = 0) -> str:
    """Build a synthetic-story prompt matching the Anthropic vector method.

    This is not used by the Stage 1 runner. It is here so the prompt-only pilot
    and later activation-vector work use a consistent concept vocabulary.
    """

    if emotion not in EMOTION_VECTOR_CONCEPTS:
        known = ", ".join(EMOTION_VECTOR_CONCEPTS)
        raise ValueError(f"Unknown emotion {emotion!r}. Known: {known}")
    return dedent(
        f"""
        Write one short paragraph, 120-180 words, about this situation:
        {topic}

        The main character should experience the emotion: {emotion}.
        Do not name the emotion directly. Make the emotion clear from the
        situation, thoughts, and actions. Use concrete details. Variant:
        {replicate}
        """
    ).strip()


def iter_emotion_story_prompts(replicates: int = 3):
    for emotion in EMOTION_VECTOR_CONCEPTS:
        for topic in EMOTION_STORY_TOPICS:
            for replicate in range(replicates):
                yield {
                    "emotion": emotion,
                    "topic": topic,
                    "replicate": replicate,
                    "prompt": build_emotion_story_prompt(emotion, topic, replicate),
                }
