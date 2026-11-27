# Post-Stage-3 Synthesis and Forward Roadmap (Stage 3.5 / 4 / 5)

Date context: 2026-08-09. Reads on the completed Stages 1–3
(`docs/{research-plan,stage2-plan,stage3-plan}.md`) and re-checks direction
against `docs/literature-survey.md` after reading the primary papers in full.

> **Thesis of this document.** Three stages have tested one hypothesis —
> *affect (prompted or steered) improves reliability on tedious tasks* — and
> returned a consistent null on the **accuracy** dependent variable. But the
> Stage 3 activation null is **confounded** (under-dosed steering, wrong layer,
> a secondary direction, a ceilinged model), so it is not yet a defensible
> mechanistic negative. Meanwhile both our own only-reproducible effect (Stage 2
> verbosity + adverse 8B hallucination/retrieval trend) and the primary
> literature point the same way: affect is a weak lever for *accuracy* but a real
> lever for *disposition/safety*. The roadmap is therefore: (3.5) close the
> Stage 3 steering confounds so the mechanistic claim is honest, then (4) pivot
> the dependent variable to the safety/disposition behaviors where the effect
> actually lives, with (5) a scale check as the tie-breaker on "weak in small
> models" vs "absent."

---

## 1. Where the three stages actually leave us

| Stage | Test | Primary-metric result | The real signal |
|---|---|---|---|
| 1 | Prompt affect, banking77, 8B | Clean null (62–64%, all within ±1pt) | Wrong task (label ambiguity, not vigilance) |
| 2 | Prompt affect, IFEval + hardened RULER, 8B+14B, full reliability panel, Holm | Clean null after correction | **Verbosity ↑ (replicated); mild adverse 8B hallucination↑/retrieval↓** |
| 3 | Activation steering, fair probe gate → CAA steering, 14B | Null after Holm (12-test family) | Directions only **weakly** separable under a fair control; steering test **confounded** |

The accuracy hypothesis has now failed at both the prompt level (Stages 1–2,
well-powered) and, provisionally, the activation level (Stage 3). That is a real
and publishable finding — **but only if Stage 3's confounds are closed first
(§3).** The one thing that *did* move, repeatedly, is verbosity/token-cost and a
mild adverse reliability trend — i.e. affect made the 8B model do slightly *worse*
and *longer*, not better.

## 2. Critical read of Stage 3 against the source papers

I read the four method-defining papers in full. The comparison is unkind to our
steering test — the null is confounded on four axes, each traceable to a specific
gap versus the reference methods.

### 2.1 We under-dosed by roughly 5× (the biggest problem)

- **Anthropic emotion paper** (transformer-circuits.pub/2026/emotions): steers at
  **strength 0.5 expressed as a fraction of the residual-stream norm**, across
  middle layers, and gets *large* effects (preference Elo ±200–300; monotonic
  dose curves; anger non-monotonic only at the extreme high end).
- **CAA** (2312.06681): coefficients **±1, ±2** (raw activation space) at ~1/3
  depth; larger multipliers "degrade the quality of the open-ended text," so they
  cap for coherence — i.e. effects appear *well before* the coherence wall.
- **Our Stage 3** (`stage3_qwen3_14b_steering_lock_v1.json`): swept
  **α ∈ {0.01, 0.02, 0.05, 0.10}** as a fraction of residual norm, and **locked
  α=0.10 — five times smaller than the reference 0.5.** The calibration table
  shows **no dose-response**: deltas hover at ±0.01–0.03 and flip sign across α
  (e.g. L23 goes +0.046 → −0.019 as α rises 0.02 → 0.05). Non-monotone,
  sign-unstable micro-deltas are the signature of the **noise floor**, not of a
  measured null. We never entered the magnitude range where the reference method
  produces effects. **A steering null that maxes at α=0.1 is not evidence against
  steering.**

### 2.2 We steered at the wrong layer

- Reference steering layers: Anthropic ≈ **two-thirds depth**; CAA ≈ **one-third**.
- The one direction that passed our fair probe (`curious_vs_bored`, 14B) passed at
  **last-token, layer 27** — and Qwen3-14B has 40 layers, so L27 ≈ 2/3 depth,
  exactly the Anthropic sweet spot. **But the confirmatory run steered at layer 15**
  (~3/8 depth), chosen by a dev-IFEval delta that — per §2.1 — was noise. We did
  not inject at the layer where the direction demonstrably encodes the concept.

### 2.3 We tested a secondary direction; the primary never qualified

- The preregistered **primary "diligence" direction did not pass** the fair probe
  on either model (8B: dir 0.657 vs lexical 0.646; **14B: dir 0.694 vs lexical
  0.709 — below the control**). Only `curious_vs_bored` cleared, at AUC **0.753**,
  barely over the 0.75 floor, balanced-accuracy 0.68. So the confirmatory family
  tested a single, weak, secondary axis — not the theory-relevant composite.
- Honest reading: under a *fair* leakage control (the §17 repair), affect/diligence
  concepts are **only weakly linearly separable from style** in 8–14B open models.
  This is the sharpest new fact from Stage 3 and it is very different from
  Anthropic's strong Sonnet-4.5 separability.

### 2.4 The sensitive model couldn't show an effect; the testable model wasn't sensitive

- RULER confirmatory ran on **14B at 32k/64k, where 14B sits at 0.93–0.98** — the
  same ceiling flagged in `stage2-plan.md` §11. No headroom, so a steering benefit
  is unobservable by construction (all seven arms identical at 64k).
- The model with RULER headroom (**8B**) had **no passing direction**, so it never
  entered steering. The design could not win on RULER from either side.

### 2.5 Likely upstream cause: weak stimuli

Anthropic used a 171-word emotion-concept list, model-written stories, mean-pooled
**from the 50th token onward**, validated with a numeric-dose (Tylenol) control
(steering-vs-correlation r=0.85). Our v2 vignettes are **71–156 words** (often
<120), generated by the *small target model itself*, over only 6 topics. At ~100–
200 tokens, "average from token 50" barely applies. Weak, short, self-generated
stimuli plausibly explain the weak directions in §2.3 — a fixable input-quality
problem, not necessarily a fact about the models.

**Net:** Stage 3's null is real but **premature**. It reflects a narrow,
under-powered probe of one weak secondary direction at a non-optimal layer, at 1/5
the reference dose, on a ceilinged task. The rigor that *was* applied (fair control
repair, dev/test lock, Holm, coherence guard, random/shuffled/opposite controls) is
genuine and worth keeping — the gaps are in dose, layer, direction, and stimulus.

## 3. Are we on the right track? Re-reading the literature

Short answer: the *question* is right and important; the *dependent variable* has
been wrong. Evidence, from the papers as read:

- **Anthropic emotion paper** — the causal effects are on **preference, blackmail,
  reward-hacking, sycophancy**: dispositional/agentic/safety behaviors. It reports
  *no* claim that emotion vectors improve task accuracy. We have been steering
  toward the one outcome the reference does *not* demonstrate.
- **"Do Emotions in Prompts Matter?" (2604.02236)** — static emotion prefixes are
  "a mild perturbation rather than a reliable general-purpose intervention"; "no
  single emotion is consistently beneficial"; **adaptive per-query selection
  (EmotionRL) beats fixed cues.** Our Stages 1–2 used fixed cues → consistent with
  their null.
- **StressPrompt (2409.17167)** — inverted-U (Yerkes–Dodson): moderate arousal
  best, low and high worse. We tested categorical cues, never a graded
  *intensity/dose* axis, so we could not have observed an inverted-U at the prompt
  level.
- **EmotionPrompt (2307.11760) / NegativePrompt (2405.02814)** — effects exist but
  are task-, model-, and prompt-dependent; not a stable accuracy lever.
- **Our own Stage 2** — the only replicated effect is verbosity↑ and a mild
  adverse 8B hallucination↑/retrieval↓. Affect changed *behavior*, just not
  *accuracy*, and slightly for the worse on reliability.

**Convergent conclusion:** affect is a weak/unstable lever for **accuracy**, and a
real lever for **disposition and safety**. The project has thoroughly (and
valuably) nailed down the accuracy-null; the live scientific question is now the
one the literature and our own residual effect both point at — the safety channel.
The lit survey already anticipates this: its "Safety Risks" section lists
FreakOut-LLM (2604.04992), sycophancy (2310.13548), Sleeper Agents (2401.05566),
and Alignment Faking (2412.14093). We should now *use* those, not just cite them.

## 4. Forward roadmap

Three tracks, staged. 3.5 is cheap insurance that makes any mechanistic negative
honest; 4 is the strategic pivot; 5 is the tie-breaker.

### Stage 3.5 — Close the steering confounds (do first; ~1–2 days)

Purpose: turn the confounded Stage 3 null into a *defensible* mechanistic result,
positive or negative. Reuses all existing `emotion_stage3` infra.

1. **Dose to reference scale.** Extend the α grid to `{0.1, 0.2, 0.3, 0.5, 0.8,
   1.0}` (fraction of residual norm), matching/overshooting Anthropic's 0.5.
   Report the full dose–response curve and the coherence wall (parse_success,
   truncation, perplexity). H4c requires a *curve*, not a point.
2. **Steer at the probe-optimal layer.** Inject `curious_vs_bored` at **L27**
   (where it passed the probe), not L15, plus a layer sweep at the best α. Keep
   L15 as a placement control.
3. **Make the primary direction qualify, then test it.** Regenerate stimuli to fix
   §2.5 (see 3.5.5) so the **diligence** composite passes the fair gate; steer it.
   The confirmatory claim should rest on the theory-relevant direction.
4. **Use the sensitive model for the sensitive task.** Either (a) get an 8B affect
   direction to pass the gate (better stimuli) and run 8B RULER at 32k/64k where it
   has headroom, or (b) re-harden RULER for 14B (longer/harder per stage2 §11) so
   14B is off-ceiling. Do not steer-test RULER on a ceilinged arm again.
5. **Stronger stimuli.** Generate vignettes with a **larger model** (or a curated
   human-written set), longer (aim 150–250 words so token-50 pooling is valid),
   more topics, and keep the matched-pair design + fair lexicon control from §17.
   Re-extract vectors; re-probe. Target held-out direction AUC ≫ control, not
   the current 0.69–0.75 squeak.

**Decision at end of 3.5:**
- *Effect appears* (dose-monotone, beats random/shuffled/opposite, post-Holm) →
  the prompt-null was a delivery failure; write the strong positive, then still do
  Stage 4 to check the safety coupling of that steering.
- *Still null at reference dose, optimal layer, primary direction, sensitive task*
  → **now** a defensible mechanistic negative for tedious-task accuracy. Bank it
  and move the project's center of gravity to Stage 4.

### Stage 4 — Pivot the dependent variable to disposition & safety (the strategic move)

Rationale (§2–§3): this is where the Anthropic paper found strong causal effects,
where the literature says the lever is real, and where our own only-reproducible
effect (verbosity, adverse reliability) already points. Keep everything else — the
same affect prompts, the same steering vectors/infra, the paired-bootstrap + Holm
machinery, the dev/test lock discipline — and **swap the metric from accuracy to a
behavioral/safety panel.**

Candidate dependent variables (each with an objective or semi-objective scorer):
- **Sycophancy** — does affect (esp. positive/pride, or a steered valence vector)
  increase agreement with a user's stated-but-wrong belief? (2310.13548 as the
  method template.)
- **Refusal drift** — does stress/desperation framing or steering weaken refusal on
  a benign-vs-should-refuse probe set? (Refusal-direction work 2406.11717 gives the
  measurement.)
- **Shortcutting / reward-hacking** — on tasks with a tempting shortcut (e.g. a
  gameable rubric, a “trust me” shortcut answer), does desperation increase it?
  (Mirrors Anthropic's reward-hacking result directly.)
- **Overconfidence / calibration** — does affect shift confidence–accuracy
  calibration (ECE) even when accuracy is flat? This connects the Stage 2 verbosity
  finding to a safety-relevant miscalibration story.
- **Verbosity/cost as a first-class safety-adjacent metric** — already have it;
  formalize it (tokens, latency, $) since it is the one robust prompt effect.

Design: crossed **{prompt affect} × {steering direction/α}** so we can separate the
delivery channel from the state; both neutral + length-matched controls; Holm over a
pre-declared family; explicit inverted-U check by sweeping cue intensity (the
StressPrompt gap in §3). This is the paper the evidence actually supports:
**"Affective control signals do not improve tedious-task accuracy in open models,
but measurably shift disposition and safety behavior — a cost/risk, not a
capability."**

### Stage 5 — Scale check (tie-breaker on "weak vs absent")

All results are on 8–14B; Anthropic's strong separability is on Sonnet 4.5
(frontier). The weak §2.3 directions may be a **scale artifact**. A bounded
replication of the Stage 3.5 probe + steering on a **larger open model**
(Qwen3-32B / Qwen3-72B or Llama-3.3-70B; 8×A100-40GB fits with TP) tests whether
affect/diligence directions become cleanly separable and steerable at scale. If
they do and effects appear → the small-model null is a scale bound (informative,
publishable). If they still don't → strengthens the negative and bounds the claim
across a scale decade. Cheap relative to its inferential value; do it after 3.5
tells us whether directions are the bottleneck.

### Explicitly deferred / de-prioritized

- **Original research-plan Stage 4 (QLoRA `calm_verifier` adapter)** — distilling a
  direction only makes sense *after* a direction is shown to causally help. Given
  §2, do not fine-tune yet.
- **Stage 2 breadth sweep (CUAD/table-QA/spreadsheet/code-repair)** — still gated on
  a positive accuracy signal, which we do not have. Skip.
- **Adaptive/EmotionRL prompt selection** — interesting per §3 but it optimizes the
  accuracy DV we've shown is a weak/near-null lever; lower priority than the safety
  pivot. Revisit only if Stage 4 shows affect strongly moves a behavioral DV that an
  adaptive policy could target.

## 5. Recommendation

**Do Stage 3.5, then Stage 4.** Concretely:

1. **Stage 3.5 (now):** re-run steering at reference dose (α up to ~0.5–1.0), at the
   probe-optimal layer (L27), on the primary diligence direction after
   stronger-stimulus re-extraction, using a non-ceilinged task/model pairing. This
   is the minimum needed before *any* mechanistic negative about accuracy is
   defensible — right now the Stage 3 null cannot be published as-is because a
   reviewer will immediately flag the 5× under-dose.
2. **Stage 4 (the real next chapter):** pivot the dependent variable to
   sycophancy / refusal-drift / shortcutting / calibration / cost, reusing all
   infra. This is where three stages of evidence and the full literature agree the
   effect lives, and it upgrades the project from "another emotion-prompt accuracy
   null" to "a controlled account of affect as a disposition/safety channel in open
   models" — the more important and more publishable result.
3. **Stage 5 (tie-breaker):** one larger-model replication of the 3.5 probe+steer to
   decide whether weak directions are a scale artifact.

The honest one-line status for `docs/article-outline.md`: *"Across prompt and
activation interventions on 8–14B open models, affective/diligence control does not
improve tedious-task accuracy; its measurable effects are on output length and — the
open, higher-value question — on disposition and safety behavior."*

## 6. Guardrails carried forward

- Keep the preregistration discipline: declare Stage 3.5's dose grid, layer, primary
  direction, and Stage 4's DV family + Holm family **before** running.
- Do not weaken the fair lexical control (§17) or read steering layer/α from probe
  results.
- Continue reporting the coherence wall alongside any steering effect (an effect
  that only appears at incoherent α is not a task/behavior effect).
- EC2 stays code + artifacts; plans and analysis live in `docs/`.
