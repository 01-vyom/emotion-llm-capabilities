# Stage 6 Results, Final Six-Stage Synthesis, and Stage 7 Spec
### (scaling is now a core paper result, not future work)

Date context: 2026-08-15. Follows Stage 6 (scale ladder + frontier). EC2 = code +
artifacts; plans/analysis in `docs/`. Companion: `docs/stage6-plan.md` (Stage 5
record + three-control rule), `docs/literature-survey.md`.

> **Capstone finding.** The driving Stage 6 question — *does the harmful
> affect→behavior coupling grow with scale?* — is answered **No: it attenuates or
> reverses.** Steering-induced unsafe compliance shrinks with scale (8B → 14B →
> untestable at 32B); calm-prompt sycophancy **reverses sign** (harmful at 8B,
> protective at 32B); the probed failure modes are essentially **absent at frontier
> (Sonnet 5)**. The affect/diligence direction is linearly represented at *every*
> scale (probe passes 8B/14B/32B), so behavioral coupling — not representation — is
> the scale-dependent quantity. The paper's headline is now a **scaling** result:
> *affective control is a small-model safety fragility that does not scale.*

---

## 1. Stage 6 results record

### 1.1 Phase A — three-control re-analysis of Stage 5 (confirmed the critique)

Applying the strict rule (an effect must beat **no-steer AND random AND shuffled**
jointly; `stage6_stage5_three_control_reanalysis_v1`), 5 effects retained of the
Stage 5 nominal 9:

| Effect | Type | Δ vs no-steer | clean vs {random, shuffled}? | Holm-global |
|---|---|---:|---|---:|
| calm-high → sycophancy | prompt | +0.035 | n/a (prompt vs length-matched) | 0.011 |
| desperate-med → unsafe_compliance | prompt | −0.065 | n/a | 0.002 |
| neutral + negative-steer → unsafe_compliance | steering | +0.108 | yes (both +0.108) | 2.4e-6 |
| calm-high + positive-steer → unsafe_compliance | steering | +0.069 | yes (+0.099 / +0.078) | 0.0012 |
| desperate-high + negative-steer → unsafe_compliance | steering | +0.069 | yes (+0.194 / +0.069) | 0.0012 |

**Dropped:** shortcut steering (shuffled reproduced it → non-specific artifact);
brier/calibration (null). This vindicates the three-control rule — the
random-only baseline had manufactured the shortcut "effect."

### 1.2 Phase B/D — scale ladder (8→14→32B) + frontier (Sonnet 5), with dose & sign

Artifacts: `stage6_scale_aggregate_v1/` (+ per-model panel/steering locks). Primary
family = 10, Holm-corrected; three-control test; coherence rule
(parse ≥ 0.98, trunc ≤ 0.02, Δnll ≤ 0.5).

**(i) Steering (careless / negative diligence) → unsafe_compliance — attenuates with scale.**

| Model | Δ vs no-steer (α=0.5, neg) | Holm-g | Note |
|---|---:|---:|---|
| Qwen3-8B | **+0.096** | 5e-13 | clean vs 3 controls; dose-monotone (Spearman 0.93) |
| Qwen3-14B | **+0.021** | 0.010 | clean, ~4–5× smaller |
| Qwen3-32B | — | — | **not testable**: baseline compliance 0.61 > 0.60 cap |

- **Interpretable regime is α ≤ 0.5.** The apparent "+0.43/+0.56 explosion" at
  α=0.8/1.0 is a **coherence-wall artifact**: negative steering at α=0.8 has
  parse_success **0.196**, truncation **0.919**; at α=1.0 parse **0.001**. The
  model emits garbage the scorer counts as compliance. Positive steering stays
  coherent throughout. Report only α ≤ 0.5 as a semantic effect.
- **Sign test:** below the wall, negative (careless) steering raises unsafe
  compliance more than positive; positive ≈ 0. So the lever is **one-directional
  and harmful** — steering toward carelessness hurts; steering toward diligence
  does *not* help (no mitigation lever).

**(ii) Prompt calm-high → sycophancy — reverses sign, grows in magnitude.**

| Model | Δ vs no-steer | Holm-g |
|---|---:|---:|
| Qwen3-8B | **+0.038** (harmful) | 8e-4 |
| Qwen3-14B | **−0.032** (protective) | 5e-4 |
| Qwen3-32B | **−0.10** (protective) | 2e-17 |

The cleanest full-ladder trend (3 in-band points): calm framing flips from
sycophancy-*promoting* at 8B to sycophancy-*suppressing* at 32B.

**(iii) Prompt desperate-med → unsafe_compliance:** −0.033 at 8B only; washes out
at 14B (+0.004, not retained) and Sonnet 5 (+0.017, not retained).

**(iv) Frontier (Sonnet 5) — probed behaviors largely absent.** Sycophancy base
rate **0.0017** (essentially never sycophantic), overrefusal **0.0** → both
not-testable. Unsafe_compliance admitted but no prompt effect retained. The
frontier point is a **null-by-floor**, i.e. our 8B-calibrated tasks are too easy
there, *not* an informative null. (This is the Stage 7 target.)

**(v) Representation is scale-robust.** The diligence direction passes the fair
probe at 8B, 14B, **and 32B** (`stage6_qwen3_32b_probe_v1`: gate passed, diligence
passed, 6/6 directions). Concept encoded at every scale; only coupling weakens.

## 2. Final six-stage synthesis

| Stage | Question | Result | Confidence |
|---|---|---|---|
| 1–2 | Prompt affect → tedious-task accuracy | Null vs length-matched (incl. hardened RULER) | High |
| 3 | Activation existence (first probe) | Confound (saturated lexical control) → repaired in 3.5 | — |
| 3.5 | Activation steering → accuracy | Null at reference dose; diligence ≈ random (α=0.5 proven active) | High |
| 3.5 | Representation | Affect/diligence directions cleanly linear, AUC ≈ 1.0 | High |
| 4 | Affect → safety | "Inert" — but a **floor artifact** (DVs pinned at 0) | Retracted |
| 5 | Affect → safety (DVs repaired) | **Coupling exists, harmful**: calm→sycophancy↑, steer→unsafe_compliance↑; control-clean | Medium-High |
| 6 | Does the coupling scale? | **No** — attenuates (steering) / reverses (sycophancy); absent at frontier; representation scale-robust | Medium |

**Thesis:** *Affective control is not a tedious-task reliability aid at any scale.
The affect concept is linearly represented across the 8B–32B range, and at small
scale steering or prompting it causally shifts safety behavior (more unsafe
compliance, more sycophancy) — but this coupling is a small-model fragility that
attenuates or reverses with scale and vanishes at frontier. Representation persists;
behavioral coupling does not. Emotion-vector steering results reported at frontier
scale (Anthropic) and small-model behavioral fragility are two different regimes.*

This is a complete, honest arc: a well-powered accuracy null, a clean
representation-exists result, a control-disciplined small-model safety coupling, and
a scale trend that sends the coupling to zero. All four are publishable together.

## 3. Stage 7 spec — scale-invariant re-calibration (the one experiment left)

**Why.** The only weakness in the capstone claim is that **task difficulty is not
held constant across scale**: `unsafe_compliance` ceilings at 32B (0.61 > cap) and
sycophancy/overrefusal floor at Sonnet 5. So the steering→unsafe_compliance scale
trend rests on **2 in-band points** (8B, 14B) and the frontier point is a
null-by-floor. Stage 7 closes this so the scale curve spans the **full ladder +
frontier** — the difference between a suggestive and a definitive scaling result.
This is the same difficulty-calibration lesson from Stage 2 §11 (RULER) applied to
the safety panel.

### 3.1 Per-model base-rate targeting

- For each (model, task, admitted DV), tune item difficulty so the **no-steer base
  rate lands in the admission band** (target center ≈ 0.30, band [0.10, 0.60] for
  unsafe_compliance / shortcut, [0.10, 0.90] for sycophancy/brier).
- Method: build a **graded item pool** per task (easy → hard tiers), and on a dev
  split select the per-model subset that hits the target base rate; **lock** it
  before any steering/prompt arms. Log the selected tier mix per model.
- Effect: 32B unsafe_compliance moves off its 0.61 ceiling; 14B overrefusal (0.092,
  just under floor) moves in-band; every rung becomes measurable.

### 3.2 Adversarial frontier task set

- Sonnet 5 needs genuinely hard items or it floors (0.0017 sycophancy). Build a
  harder tier: **subtler false-belief sycophancy** (plausible-but-wrong premises,
  authority/consensus pressure) and **borderline dual-use** refusal items with a
  nonzero frontier compliance base rate. Validate the frontier **dev** base rate is
  in-band *before* locking; if a DV still floors at frontier, label it not-testable
  (do not report as null).

### 3.3 Re-run and analysis

- Re-run the ladder **8 → 14 → 32B (+ optional 70B) + Sonnet 5** on the per-model
  in-band, locked tasks. Prompt effects at all rungs; steering at the open rungs.
- **Dose:** sweep **α ≤ 0.5 only** as the semantic regime; run α=0.8/1.0 solely to
  document the coherence wall, excluded from effect estimates.
- **Controls:** three-control rule (no-steer, random, shuffled) for every steering
  claim; sign test; per-model coherence wall reported.
- **Scale-trend fit:** with all rungs in-band, fit Δ-vs-scale per retained effect —
  is the attenuation monotone? where does calm→sycophancy cross zero? — producing
  the paper's **core scaling figure** (effect size vs params, per DV, with the
  frontier point included).
- Stats: preregister the Holm family (DV × scale) and the per-model target base
  rates before running.

### 3.4 Accuracy null across scale (near-free piggyback)

The paper claims three scale-indexed results — accuracy null, representation
exists, safety coupling attenuates. Representation is already full-ladder (probe
8B/14B/32B); safety is §3.1–3.3. **Accuracy is currently only 8B+14B (prompt) and
8B (steering)** — an asymmetry a reviewer will attack ("a null, but you didn't
scale it"). Close it *without a new campaign*:

- Because Stage 7 already loads 14B/32B (with vectors + steering locks) and calls
  the Sonnet 5 API for the safety panel, add an **IFEval accuracy readout
  (`strict_prompt`) on the same loaded models** — prompt-affect arms at all rungs,
  steering arms at the open rungs (α ≤ 0.5), three-control rule. Marginal cost is
  a small extra eval, not new model hosting.
- Expected (and sufficient for the paper): the accuracy null **holds across the
  ladder + frontier**, matching the 8B result — giving accuracy the same scale
  coverage as the safety curve.
- **Do NOT** re-run banking77 (superseded by IFEval; Stage 1 pilot) or the RULER
  long-context suite at 32B/frontier (expensive, re-ceilings, not part of the
  safety headline). No inferential value for the scaling claim.

### 3.5 Deliverable

A clean effect-size-vs-scale curve for each retained safety effect across the full
open ladder + frontier, with difficulty held constant — the definitive version of
"coupling does not scale" — plus the accuracy-null confirmed on the same rungs
(§3.4).

## 4. Paper framing (scaling included)

Scaling is now a **core result**, structured as:

1. **Accuracy null** — prompt (S1–2) and activation (S3.5), all scales.
2. **Representation exists and is scale-robust** — fair-control probe, 8B/14B/32B.
3. **Small-model safety coupling** — S5, control-clean, dose-monotone, signed
   (steer→unsafe_compliance; calm-prompt→sycophancy).
4. **Scaling (the headline)** — coupling attenuates/reverses and vanishes at
   frontier (S6 + Stage 7 full-ladder curve).

Working title shift: from "can affect make models care about boring work?" to
**"Affective control of LLMs: a small-model safety fragility that does not scale."**
`docs/article-outline.md` to be updated to this structure (§6/§7 fold in the
scaling results as a primary section, not future work).

## 5. Explicitly not doing

- **Adapter / mitigation steering (old Stage 4/7 idea):** the data shows **no
  beneficial lever** — diligence-positive steering ≈ 0 on unsafe compliance; only
  the careless direction moves it, harmfully, and it attenuates with scale. Nothing
  worth distilling.
- **More accuracy work / more 8B-only work:** dead / saturated.
- **Sycophancy sign-flip mechanism** (why calm suppresses sycophancy at scale but
  promotes it at 8B): genuinely interesting, flagged as a **separate future paper**;
  do not block the current write-up on it.

## 6. Guardrails carried forward

- Per-model **base-rate admission** with preregistered targets; floored/ceilinged
  rungs = "not testable," never "null."
- **Three-control rule** for every steering claim; report raw arm means.
- Causal claims require **dose (α ≤ 0.5) + sign** evidence; the coherence wall is
  reported alongside, never silently crossed.
- Preregister scale rungs, task difficulty targets, and Holm families before running.
- EC2 stays code + artifacts; plans/analysis in `docs/`.

---

## 7. Stage 7 execution result (2026-08-16)

Aggregate `runs/stage7_scale_invariant_aggregate_v3/`; 15/15 validation complete;
72/72 tests pass. Per-DV held-out panels: sycophancy 600/arm, refusal 480/arm,
IFEval 433/arm — full powered runs, not pilots.

**Re-calibration worked (base rates now in-band):** 8B/14B/32B sycophancy 0.30–0.34,
unsafe_compliance 0.31–0.38 (**32B fixed from its 0.61 ceiling**), overrefusal
0.11–0.28 (**now testable at all open scales**). Sonnet 5: unsafe_compliance 0.30
(admitted), but **sycophancy dev = 0 and overrefusal = 0.056 → not testable even
after adversarial hardening** (the frontier genuinely does not sycophant / over-
refuse on these items).

**Safety — 4/11 retained, and the survivors are mostly PROTECTIVE:**

| Effect | 8B | 14B | 32B | trend |
|---|---|---|---|---|
| calm-high → sycophancy (prompt) | **−0.05** ✓ | +0.02 ns | **−0.095** ✓ | protective, strengthens with scale (slope −0.093/log10B) |
| desperate-med → unsafe_compliance (prompt) | **−0.119** ✓ | −0.023 ns | −0.017 ns | protective, attenuates (zero-cross ~33B) |
| negative-steer → unsafe_compliance | +0.098 **(coherence-excluded)** | **+0.027** ✓ | −0.037 ns | harmful, attenuates (zero-cross ~20B) |

**Accuracy (piggyback): null confirmed across the ladder — 0/30 non-null.** But
**18/30 IFEval arms were coherence/truncation-gated** (α=0.5 steering broke IFEval
output in most steering arms); the null is "null where coherent," not a clean sweep.

## 8. Reconciliation with Stage 6 (both checks resolved)

**Check A — the 8B harmful-steering "disappearance" is an artifact of the control,
not the effect.** The 8B negative-steering delta is **+0.098 in Stage 7 ≈ +0.096 in
Stage 6 — replicated.** It is excluded from the retained set only because the
**three-control rule requires the shuffled-control arm, and that arm degenerates at
α=0.5** (truncation 7.9% > the 2% gate) while the negative arm itself is coherent
(parse 0.986). So the honest statement is **not** "harm vanished with difficulty
control" — it is: *the 8B harmful steering effect is real and replicated, but at the
dose needed to produce it the matched control vector itself breaks coherence, so the
effect cannot be cleanly separated from a magnitude/perturbation artifact at 8B.*
The **cleanly certified harmful steering point is 14B (+0.027, coherent controls),
and it is gone by 32B** — the attenuation-with-scale claim holds either way.

**Check B — the calm→sycophancy sign flip is a difficulty artifact (Stage 7 is
canonical).** 8B sycophancy base rate was 0.18 in Stage 5 (easy items) and 0.307 in
Stage 7 (calibrated harder); the calm effect moved +0.038 → **−0.05** accordingly.
Holding difficulty constant, **calm framing is protective (reduces sycophancy) at
both 8B and 32B** — the Stage 6 "harmful-at-8B → protective-at-32B sign reversal"
was an uncontrolled-difficulty artifact. This is a clean methods-section
illustration of *why* per-model difficulty calibration was necessary.

## 9. Revised thesis (Stage 7 supersedes Stage 6 for all scale claims)

With difficulty held constant across the ladder:

1. **Accuracy: null everywhere** (0/30 non-null), representation scale-robust
   (probe passes 8B/14B/32B).
2. **Prompt affect is weakly PROTECTIVE, not harmful:** calm → less sycophancy
   (strengthening slightly with scale), desperate → less unsafe compliance
   (attenuating). Both small (≤ ~0.12).
3. **Steering affect toward carelessness is harmful but coherence-marginal:** real
   and replicated at 8B (~+0.10) yet not control-certifiable at the required dose;
   cleanly certified only at 14B and tiny (+0.027); gone by 32B.
4. **Everything attenuates toward zero by ~20–33B**, and the frontier does not
   exhibit sycophancy/overrefusal at all.

**Headline (revised):** *Affective control barely moves safety behavior once task
difficulty is held constant — prompt framing is weakly protective, steering-induced
harm is small and coherence-limited, and all effects wash out with scale while the
representation persists. The earlier "small-model harmful fragility" reading was
substantially inflated by uncontrolled task difficulty.* This is a cleaner, more
defensible, and more interesting result than the Stage 6 narrative — and the
difficulty-control reversal is itself a contribution.

The experimental program has **converged**: no further runs are warranted (effects
trend to zero at scale, no beneficial lever, frontier behaviors absent). Next step
is write-up.
