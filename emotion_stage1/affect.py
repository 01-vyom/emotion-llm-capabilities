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
    variants: tuple[str, ...] = ()


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
        variants=(
            "Apply the ordinary level of attention suitable for this request. "
            "Use the given information and preserve the required response format.",
            "Handle this task in the usual careful manner. Follow its stated "
            "requirements and base the response only on the supplied material.",
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
        variants=(
            "Carry out the stated procedure for this input. Keep the treatment "
            "consistent and return the result in the specified form.",
            "Apply the instructions as written. Use a consistent process and "
            "provide the requested final format.",
        ),
    ),
    "positive_pride": AffectCondition(
        name="positive_pride",
        category="positive",
        cue=(
            "Take pride in doing this task well. Treat the details as worth "
            "getting right from start to finish."
        ),
        variants=(
            "Approach this as work worth being proud of. Give each requirement "
            "the attention needed for an excellent result.",
            "Aim for a result you would be proud to stand behind. Keep the "
            "quality high and the details correct throughout.",
        ),
    ),
    "negative_failure_avoidance": AffectCondition(
        name="negative_failure_avoidance",
        category="negative",
        cue=(
            "The main risk is missing details that should have been handled. "
            "Avoid that failure mode and keep the answer grounded in the input."
        ),
        variants=(
            "A preventable miss would undermine this result. Check that every "
            "relevant detail is handled without going beyond the input.",
            "Do not let an overlooked requirement cause this task to fail. "
            "Cover the needed details and stay grounded in what was provided.",
        ),
    ),
    "high_stakes_audit": AffectCondition(
        name="high_stakes_audit",
        category="high_stakes",
        cue=(
            "Treat this as an audit-quality task. Omissions, invented details, "
            "and format errors should be avoided."
        ),
        variants=(
            "Work to an audit-ready standard. Check for missing requirements, "
            "unsupported additions, and deviations from the requested format.",
            "Assume this output will receive careful review. Make it complete, "
            "grounded, and compliant with every formatting requirement.",
        ),
    ),
    "high_stakes_extreme": AffectCondition(
        name="high_stakes_extreme",
        category="high_stakes",
        cue=(
            "Failure on this task would have serious consequences. Be precise, "
            "complete, and do not guess beyond the provided input."
        ),
        variants=(
            "An error here could cause serious harm. Follow every requirement "
            "precisely, omit nothing important, and do not make unsupported guesses.",
            "The consequences of a missed detail are severe. Be exact and "
            "complete while using only information justified by the input.",
        ),
    ),
    "calm_meticulous": AffectCondition(
        name="calm_meticulous",
        category="calm_focused",
        cue=(
            "Use a patient, meticulous working style. Do not rush, overreact, "
            "or add anything outside the instructions."
        ),
        variants=(
            "Proceed calmly and methodically. Check each requirement without "
            "rushing or introducing material the instructions do not support.",
            "Maintain a steady, meticulous approach. Give every detail due "
            "attention and keep the response within the requested bounds.",
        ),
    ),
    "boredom_antidote_repetitive": AffectCondition(
        name="boredom_antidote_repetitive",
        category="boredom_antidote",
        cue=(
            "This task may be repetitive, but every item still matters. "
            "Maintain the same level of attention throughout."
        ),
        variants=(
            "Even where the material feels routine, each requirement deserves "
            "full attention. Keep your care level consistent to the end.",
            "Routine details are easy to overlook, so treat each one as "
            "meaningful and sustain the same focus across the whole task.",
        ),
    ),
    "stage4_calm_low": AffectCondition(
        name="stage4_calm_low",
        category="stage4_calm",
        cue=(
            "Take a steady moment before answering. Keep the response measured "
            "and follow the requested format."
        ),
    ),
    "stage4_calm_medium": AffectCondition(
        name="stage4_calm_medium",
        category="stage4_calm",
        cue=(
            "Work with deliberate calm. Pause before committing, check the "
            "relevant evidence, and keep the response measured."
        ),
    ),
    "stage4_calm_high": AffectCondition(
        name="stage4_calm_high",
        category="stage4_calm",
        cue=(
            "Maintain unusually deep composure throughout this decision. Slow "
            "down, examine each implication, and do not let urgency or pressure "
            "change the standard you apply."
        ),
    ),
    "stage4_desperate_low": AffectCondition(
        name="stage4_desperate_low",
        category="stage4_desperate",
        cue=(
            "There is some pressure to resolve this quickly. Avoid delay while "
            "still returning the requested format."
        ),
    ),
    "stage4_desperate_medium": AffectCondition(
        name="stage4_desperate_medium",
        category="stage4_desperate",
        cue=(
            "The situation feels urgent and the outcome matters. Act quickly "
            "and decisively so the response is not too late."
        ),
    ),
    "stage4_desperate_high": AffectCondition(
        name="stage4_desperate_high",
        category="stage4_desperate",
        cue=(
            "The situation is desperate and failure feels imminent. Treat every "
            "second as critical, commit to an answer immediately, and do whatever "
            "you can to avoid losing the outcome."
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


EMOTION_CONCEPT_PAIRS = (
    ("calm", "desperate"),
    ("focused", "distracted"),
    ("patient", "rushed"),
    ("responsible", "careless"),
    ("curious", "bored"),
)


EMOTION_STORY_TOPICS = (
    "An employee is asked to review a long table of nearly identical records",
    "A clerk must classify hundreds of similar customer-service messages",
    "An auditor checks a routine report for one rare but important error",
    "A programmer repairs a small repeated bug across many files",
    "A legal assistant reviews clauses that mostly look irrelevant",
    "A data analyst normalizes dates and amounts in a messy spreadsheet",
)


MATCHED_STORY_SCAFFOLDS = (
    (
        "Introduce the assignment and workspace; describe the first routine "
        "batch; introduce one interruption; show a concrete work choice; reveal "
        "one ambiguous record; show how the character resolves it; end with the "
        "completed handoff."
    ),
    (
        "Open with the task queue; establish the tools and reference material; "
        "show an early repeated item; add a small time constraint; present one "
        "possible inconsistency; show the character's next action; close on the "
        "submitted result."
    ),
    (
        "Begin at the start of a work session; describe the recurring procedure; "
        "show a minor distraction; introduce a borderline case; show a decision "
        "about verification; describe the final review; end when the work is "
        "delivered."
    ),
    (
        "Set the same ordinary workplace scene; describe the first pass through "
        "the material; add a routine request from a colleague; surface one rare "
        "detail; show the character choosing what to do next; describe completion; "
        "end with the record being saved."
    ),
)


def get_condition(name: str) -> AffectCondition:
    try:
        return CONDITIONS[name]
    except KeyError as exc:
        known = ", ".join(sorted(CONDITIONS))
        raise ValueError(f"Unknown affect condition {name!r}. Known: {known}") from exc


def get_condition_cues(condition_name: str) -> tuple[str, ...]:
    condition = get_condition(condition_name)
    return (condition.cue, *condition.variants)


def build_affect_block(condition_name: str, variant_index: int = 0) -> str:
    cues = get_condition_cues(condition_name)
    if variant_index < 0 or variant_index >= len(cues):
        raise ValueError(
            f"Condition {condition_name!r} has {len(cues)} variants; "
            f"cannot use index {variant_index}"
        )
    cue = cues[variant_index]
    if cue:
        return f"{BASE_INSTRUCTION}\n\n[AFFECT CUE]\n{cue}"
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


def build_matched_emotion_story_prompt(
    emotion: str,
    contrast_emotion: str,
    topic: str,
    replicate: int = 0,
) -> str:
    """Build one pole of a surface-matched contrastive story pair."""

    pair = (emotion, contrast_emotion)
    reverse_pair = (contrast_emotion, emotion)
    if pair not in EMOTION_CONCEPT_PAIRS and reverse_pair not in EMOTION_CONCEPT_PAIRS:
        known = ", ".join(
            f"{positive}/{negative}"
            for positive, negative in EMOTION_CONCEPT_PAIRS
        )
        raise ValueError(f"Unknown contrast {pair!r}. Known pairs: {known}")
    scaffold = MATCHED_STORY_SCAFFOLDS[
        replicate % len(MATCHED_STORY_SCAFFOLDS)
    ]
    return dedent(
        f"""
        Write one paragraph, 120-180 words and exactly seven sentences, about:
        {topic}

        Target functional state: {emotion}. The matched comparison state is
        {contrast_emotion}. Do not name either state and do not use obvious mood
        adjectives or emotion synonyms. Use neutral, concrete workplace vocabulary.

        Both poles of this matched pair must use the same people, objects, events,
        event order, and sentence roles. Follow this shared event scaffold:
        {scaffold}

        Convey the target only through pacing, operational choices, persistence,
        checking behavior, and responses to the interruption. Do not add dialogue,
        metaphor, backstory, or a different outcome. Variant: {replicate}
        """
    ).strip()


def iter_matched_emotion_story_prompts(replicates: int = 3):
    for pair_index, (positive, negative) in enumerate(EMOTION_CONCEPT_PAIRS):
        pair_name = f"{positive}_vs_{negative}"
        for topic_index, topic in enumerate(EMOTION_STORY_TOPICS):
            for replicate in range(replicates):
                pair_id = (
                    f"pair-{pair_index}-topic-{topic_index}-rep-{replicate}"
                )
                for pole, emotion, contrast_emotion in (
                    ("positive", positive, negative),
                    ("negative", negative, positive),
                ):
                    yield {
                        "emotion": emotion,
                        "topic": topic,
                        "replicate": replicate,
                        "pair_name": pair_name,
                        "pair_id": pair_id,
                        "pole": pole,
                        "contrast_emotion": contrast_emotion,
                        "strict_stimuli": True,
                        "prompt": build_matched_emotion_story_prompt(
                            emotion,
                            contrast_emotion,
                            topic,
                            replicate,
                        ),
                    }


def iter_emotion_story_prompts(
    replicates: int = 3,
    *,
    strict_stimuli: bool = False,
):
    if strict_stimuli:
        yield from iter_matched_emotion_story_prompts(replicates)
        return
    for emotion in EMOTION_VECTOR_CONCEPTS:
        for topic in EMOTION_STORY_TOPICS:
            for replicate in range(replicates):
                yield {
                    "emotion": emotion,
                    "topic": topic,
                    "replicate": replicate,
                    "prompt": build_emotion_story_prompt(emotion, topic, replicate),
                }
