# Article Outline

Working title:

**Can We Make Language Models Care About Boring Work?**

Subtitle:

**Functional emotion as a control signal for tedious-task reliability**

## 1. Opening

LLMs are usually tested on salient tasks: math, coding, exams, reasoning,
creative writing. But much useful work is boring. It involves checking every
row, extracting every field, preserving schema, classifying many similar
messages, or finding one rare error in a long routine document.

The question is not whether models feel boredom or stress. The question is
whether affective framing or internal affect-like representations can change
the model's reliability profile on this kind of work.

## 2. Research Philosophy

Use Hamming's standard: work on an important problem with a plausible attack.

Important problem:

LLMs will be deployed as clerks, auditors, assistants, and code-maintenance
workers. Silent failures on boring tasks may matter more than spectacular
failures on benchmark puzzles.

Plausible attack:

Prior work shows emotional prompts can change benchmark performance, and the
Anthropic emotion paper shows internal emotion-concept representations can
causally influence model behavior.

## 3. Prior Work

Cover four threads:

1. EmotionPrompt, NegativePrompt, StressPrompt.
2. The 2026 Anthropic functional-emotion paper.
3. Human boredom, vigilance, and stress as careful analogy.
4. Representation engineering and activation steering.

Key caveat:

Do not anthropomorphize. Use "affective framing" and "functional emotion-like
control signals."

## 4. Hypotheses

H1:

Moderate responsibility or audit-style framing improves tedious-task accuracy
relative to neutral prompts.

H2:

Extreme stress underperforms moderate salience, especially on safety and
shortcutting metrics.

H3:

Affective prompts only count as evidence if they beat length-matched neutral
and generic diligence controls.

H4:

If affect-like activation vectors mediate the effect, then steering those
vectors should reproduce or reverse the prompt effect.

## 5. Stage 1 Experiment

Dataset:

`PolyAI/banking77`, a repetitive fine-grained banking intent classification
benchmark.

Model:

`Qwen/Qwen3-8B`, current open-weight 8B-scale model suitable for one A100.

Conditions:

- neutral
- length-matched neutral
- positive pride
- failure avoidance
- audit stakes
- extreme stakes
- calm meticulous
- boredom-antidote

Metrics:

- accuracy
- macro-F1
- parse rate
- bootstrap confidence intervals
- confusion counts

Expected interpretation:

If affective cues help only against the empty baseline, the result is weak.
If they beat length-matched neutral and generic diligence controls, the result
is worth scaling.

## 6. Stage 2 Extension

Broaden to:

- instruction compliance
- long-context retrieval
- contract extraction
- table QA
- spreadsheet tasks
- code repair

Analyze:

- omissions
- false positives
- schema errors
- position-wise decay
- token cost
- format-valid wrong answers

## 7. Stage 3 Mechanistic Test

Use Anthropic-inspired synthetic emotion stories to extract directions:

- calm vs desperate
- focused vs distracted
- patient vs rushed
- responsible vs careless
- curious vs bored

Then run:

- prompt x vector crossed design
- dose-response sweep
- layer sweep
- random and shuffled controls
- opposite-sign steering

## 8. Safety Results

Report performance and harm together.

Safety metrics:

- sycophancy
- unsafe compliance
- refusal drift
- hallucination
- overconfidence
- reward-hacking or shortcutting
- verbosity and cost

Extreme stress should be treated as a safety stress test, not an applied
recommendation.

## 9. Possible Outcomes

Outcome A:

Moderate affective framing improves tedious-task reliability without safety
regression. This supports scaling to activation steering and more tasks.

Outcome B:

Length-matched neutral controls perform equally well. The honest conclusion:
emotion is not the active ingredient; salience or instruction strength is.

Outcome C:

Stress improves accuracy but worsens safety. The article becomes a tradeoff
paper: pressure helps until it becomes a safety bug.

Outcome D:

No reliable effect. The contribution is a negative result with a reusable
benchmark and control framework.

## 10. Claim Discipline

Do not claim:

- LLMs feel boredom
- stress makes models smarter
- emotional prompting is generally useful

Claim only what survives controls:

Moderate affective or responsibility framing may shift open models toward more
reliable behavior on some tedious tasks, but effects are task-specific,
model-specific, and safety-sensitive.
