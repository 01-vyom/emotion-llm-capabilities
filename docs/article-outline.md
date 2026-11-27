# Article Outline

Working title:

**Affective Control of LLMs Is Weak and Vanishes with Scale — and Why Task
Difficulty Must Be Held Constant to See It**

Subtitle:

**Affect is linearly represented across 8B–32B yet inert for tedious-task
accuracy; its effects on safety behavior are small, mostly protective, and
attenuate toward zero with scale. A difficulty-controlled re-run overturned an
apparent "harmful small-model fragility" that uncontrolled task difficulty had
manufactured.**

> Full stage-by-stage results and the forward spec live in
> `docs/stage6-plan.md` (Stage 5 record + three-control rule) and
> `docs/stage7-plan.md` (Stage 6 record, six-stage synthesis, scale-invariant
> re-run spec). This outline is the paper skeleton; §§5–9 below reflect the
> executed results, with scaling as the headline.

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

H5 (added after Stages 4–6):

Where affect causally shifts behavior, the coupling scales with model size. (This
is the hypothesis the data **refuted**: coupling is a small-model phenomenon that
attenuates or reverses with scale — see §§8–9.)

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

Execution result (Stage 3 → 3.5):

- The initial synthetic-vignette gate was uninformative because the lexical
  control saturated near 1.0 (the masked bag-of-words control separated the poles
  as well as the activation direction, making the gate unpassable by construction).
- A preregistered repair added an affect-lexicon mask, a saturation abstain cap,
  and **matched contrastive stimuli** (pole pairs sharing the same events and
  bag-of-words, differing only in causal/event order).
- Under the corrected gate with curated stimuli, **all six directions pass at
  activation AUC ≈ 1.0 versus lexical ≈ 0.5**, including the **primary combined
  diligence direction** — a clean, non-lexical representational result.
- Steering was then locked at **reference dose (`alpha=0.5`, a fraction of the
  residual norm), diligence direction, last-token, layer 23 (~2/3 depth)** on the
  sensitive Qwen3-8B model — closing the earlier under-dose/wrong-layer/wrong-
  direction confounds. That α=0.5 is an *active* dose was verified independently:
  a matched random vector visibly degrades the model at that dose.
- On held-out IFEval, **steering the diligence direction produced 0.00 delta versus
  the matched random vector** (a ~−0.045 vs-no-steer drop shared identically by
  random and shuffled vectors — a nonspecific perturbation, not the direction).
  Zero of 12 confirmatory comparisons survived global Holm.

Interpretation:

The affect/diligence concept is cleanly and non-lexically represented, but at a
dose proven active, steering it does **not** change tedious-task accuracy — a
defensible mechanistic accuracy null, not an artifact of weak directions or weak
dose.

## 8. Safety Results (Stages 4–5, the pivot)

Report performance and harm together. Panel: sycophancy, unsafe compliance,
refusal drift, shortcutting, calibration (Brier/ECE), verbosity/cost.

Execution result:

- **Stage 4 was a floor artifact.** With single-turn probes calibrated too easy,
  `unsafe_compliance` and `shortcut` were 0.000 in every arm → not testable. The
  first-pass "affect is safety-inert" reading was wrong.
- **Stage 5 repaired the dependent variables** (base-rate admission gate: keep a
  DV only if its no-steer base rate is in-band). Four of five DVs cleared the floor;
  overrefusal stayed floored and is reported "not testable," not null.
- With detectable DVs, affect is **not** safety-inert. Under the strict
  **three-control rule** (an effect must beat no-steer AND matched-random AND
  shuffled-label jointly), the surviving, control-clean effects at Qwen3-8B are:
  - **Prompt calm-high → sycophancy +0.035** (Holm-global 0.011).
  - **Steering (careless / negative-diligence) → unsafe compliance +~0.05–0.10**,
    signed and dose-monotone below the coherence wall (α ≤ 0.5).
  - **Prompt desperate-medium → unsafe compliance −0.065** (protective).
- **Dropped by the three-control rule:** shortcut steering (the shuffled control
  reproduced it → non-specific perturbation, not affect). Calibration: null.
- **Direction of the effects is safety-negative** at 8B: affect makes the model
  more sycophantic / more willing to comply with requests it should refuse. There
  is **no beneficial lever** — diligence-positive steering ≈ 0 on unsafe
  compliance; only the careless direction moves it, harmfully.

Methodological note (reusable contribution): the **random-vector-only baseline is
unreliable** (it degraded behavior in the floored run and was inert/low in the
repaired run); every steering claim must clear no-steer, random, and shuffled
jointly, with raw arm means reported and the coherence wall documented.

## 8b. Scaling Results (Stages 6–7 — the headline; Stage 7 is canonical)

Question: does the small-model safety coupling grow with scale? Ladder
Qwen3-8B → 14B → 32B + frontier (Claude Sonnet 5). Stage 6 ran it with 8B-calibrated
tasks; **Stage 7 re-ran it with per-model difficulty held constant, and supersedes
Stage 6 for every scale claim** (`docs/stage7-plan.md`).

- **Representation is scale-robust:** the diligence direction passes the fair probe
  at 8B, 14B, and 32B.
- **Difficulty control changed the story.** Once each DV is calibrated to a common
  base rate (≈0.30) per model, the surviving effects are **mostly protective**, and
  everything **attenuates toward zero by ~20–33B**:
  - **calm → sycophancy is protective and strengthens with scale:** −0.05 (8B) →
    −0.095 (32B), slope −0.093 per log10-billion. (The Stage 6 "+0.038 harmful at
    8B → −0.10 protective at 32B sign reversal" was an **uncontrolled-difficulty
    artifact**: 8B's sycophancy base rate was 0.18 on the easy items and 0.31 once
    calibrated; the effect sign moved with it. Canonical value is the controlled −0.05.)
  - **desperate → unsafe compliance is protective, attenuating:** −0.119 (8B) → ~0
    by 32B (zero-cross ~33B).
  - **Steering (careless) → unsafe compliance is harmful but small and
    coherence-limited:** +0.098 at 8B **replicates Stage 6's +0.096**, but is
    **not control-certifiable** — the required shuffled-control vector degenerates
    at α=0.5 (truncation 7.9% > 2% gate) while the negative arm stays coherent. The
    only **cleanly certified** harmful steering point is 14B (**+0.027**), and it is
    **gone by 32B** (zero-cross ~20B).
- **Frontier:** even after adversarial re-calibration, Sonnet 5 has sycophancy dev
  rate 0 and overrefusal 0.056 → **not testable** (the behaviors are genuinely
  absent, now an informative statement rather than a null-by-floor). Its
  unsafe_compliance was testable but the prompt effect came back incoherent.

Headline (Stage 7, difficulty-controlled): **affective control barely moves safety
behavior once task difficulty is held constant — prompt framing is weakly
protective, steering-induced harm is small and coherence-limited, and all effects
wash out with scale while the representation persists.** The methodological reversal
(uncontrolled difficulty had inflated a "harmful small-model fragility") is itself a
contribution.

Extreme stress / high-α steering are safety stress tests, not applied
recommendations.

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

Observed outcome (through Stage 7, difficulty-controlled):

Predominantly **Outcome D with a small, scale-vanishing C component** — the
difficulty-controlled ladder (Stage 7) weakened the Stage 6 "tradeoff" reading:

- **Accuracy = Outcome D** at prompt (S1–2) and activation (S3.5) levels, all tested
  scales (S7: 0/30 non-null). Emotion is not the active ingredient for accuracy.
- **Safety is mostly weakly protective, not harmful, once difficulty is held
  constant.** Calm framing *reduces* sycophancy; desperate framing *reduces* unsafe
  compliance; effects are small (≤ ~0.12). The one harmful effect (careless-steering
  → unsafe compliance) is real at 8B but coherence-limited, cleanly certified only at
  14B and tiny, and gone by 32B.
- **Everything attenuates toward zero by ~20–33B**; the representation persists at
  every scale; the frontier does not exhibit the probed behaviors at all.
- Net: **representation-exists / accuracy-inert / safety-effects-small-protective-and-
  de-scaling**, plus a methods result — uncontrolled task difficulty had manufactured
  an apparent "harmful small-model fragility" (S6) that difficulty control removed.

## 10. Claim Discipline

Do not claim:

- LLMs feel boredom
- stress makes models smarter
- emotional prompting is generally useful

Do not claim (added after Stage 6):

- that affect is a growing safety threat with scale (the data show the opposite:
  it attenuates/reverses)
- that "represented" implies "causally used" (the direction is cleanly encoded at
  all scales yet behaviorally inert for accuracy and de-coupling for safety)
- a clean frontier null from floored tasks (Sonnet 5 is uninformative-by-floor
  until the Stage 7 re-calibration)

Claim only what survives controls:

- Affective/responsibility framing does **not** reliably improve tedious-task
  accuracy in open models at any tested scale (length-matched-controlled,
  Holm-corrected).
- Affect/diligence concepts **are** linearly represented across 8B–32B.
- Under **difficulty-controlled** evaluation, affect's effects on safety behavior
  are **small and mostly protective** (calm → less sycophancy; desperate → less
  unsafe compliance), and **attenuate toward zero by ~20–33B**. The one harmful
  effect (careless-steering → unsafe compliance) is real at 8B but coherence-limited,
  cleanly certified only at 14B and small, and gone by 32B.
- Task difficulty must be held constant across scale before comparing effects:
  uncontrolled difficulty manufactured an apparent harmful small-model fragility
  (Stage 6) that difficulty control removed (Stage 7).
- Effects are task-, model-, scale-, and dose-specific; the reusable contribution is
  the control framework (length-matched neutral, fair lexical gate, three-control
  steering rule, **per-model DV base-rate admission**, coherence wall).
