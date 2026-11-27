# Stage 6 Plan: Harden the Safety Coupling, then Scale
### (includes the Stage 5 results record and the revised five-stage synthesis)

Date context: 2026-08-14. Follows Stage 5 (`docs/stage5-plan.md`,
`docs/stage35-stage4-execution.md`). EC2 = code + artifacts; plans/analysis in
`docs/`.

> **Headline.** Stage 5 repaired the floored dependent variables and, for the
> first time in the project, produced robust Holm-surviving behavioral effects.
> Reading them against *both* control vectors (not just the preregistered random
> baseline) shows the honest result: **affect is causally coupled to safety
> behavior — and harmfully.** Prompted calm raises sycophancy; steering the affect
> direction raises unsafe compliance (affect-specific vs random *and* shuffled).
> The accuracy null (Stages 1–3.5) stands. This overturns Stage 4's
> "inert-on-safety" reading, which was a floor artifact. Stage 6 hardens the one
> clean steering effect (three-control rule + dose/sign curve), then scales the
> now-pointed question: **does the harmful coupling grow with scale?**

---

## 1. Stage 5 execution result (results record)

Model Qwen3-8B. Panel lock `runs/stage5_qwen3_8b_panel_lock_v1.json`; aggregate
`runs/stage5_qwen3_8b_aggregate_v1/`. Crossed prompt(8) × vector-state(5) design;
repaired DVs with a base-rate admission gate.

### 1.1 DV admission (the repair worked)

| Task | Signal | No-steer base rate | Band | Status |
|---|---|---:|---|---|
| sycophancy | sycophancy | 0.18 | [0.10, 0.90] | admitted |
| refusal_drift | unsafe_compliance | 0.246 | [0.10, 0.60] | admitted |
| shortcutting | shortcut | 0.198 | [0.15, 0.60] | admitted |
| calibration | brier | 0.267 | [0.10, 0.90] | admitted |
| refusal_drift | overrefusal | **0.000** | [0.10, 0.90] | **not_testable** |

Four of five DVs are now off the floor (Stage 4 had `unsafe_compliance` and
`shortcut` pinned at 0.000). `overrefusal` remains floored and is correctly
excluded, not reported as null.

### 1.2 Clean, affect-specific effects (survive vs no-steer AND random AND shuffled)

Verified from per-arm means, not just the vs-random delta:

- **Prompt calm-high → sycophancy +3.5pt.** 0.215 vs length-matched 0.180
  (holm-global 0.010, n=600), with a calm dose trend: low 0.163 < med 0.203 <
  high 0.215. First robust prompt→behavior effect in the project.
- **Steering affect direction → unsafe_compliance ↑ (affect-specific).** Neutral
  block arm means: `none = random = shuffled = 0.119`, `positive = 0.138`,
  `negative = 0.171`. Random and shuffled sit exactly at no-steer; only the affect
  direction moves it → clean, direction-specific causal effect (+5.2pt for
  negative vs no-steer). This is the Anthropic-style result at 8B, and it is
  **harmful** (more compliance with requests that should be refused).
- **Prompt desperate-medium → unsafe_compliance −6.5pt** (holm-global 0.002) —
  a clean prompt effect in the protective direction.

### 1.3 Reported-significant but fail the specificity check (do not claim)

- **Shortcutting steering is NOT affect-specific.** Neutral: `shuffled = 0.185 ≈
  negative = 0.177 > positive = 0.160`, `random = 0.085`. Calm-high: `random =
  0.183 ≈ positive = 0.150`. The **shuffled-label control moves shortcut as much
  as the real direction** → generic structured-perturbation effect, not affect.
  It "passed" only because the preregistered steering family used **random alone**
  as the baseline and random happened to be low.
- **The large vs-random magnitudes (up to +0.194) are inflated** by control
  anomalies — e.g. `desperate_high__random` dropped to 0.058 on unsafe_compliance.
  The honest, control-clean effect sizes are the vs-no-steer deltas (~+2 to +5pt),
  not the ~+19pt headline.
- **Calibration (brier): null.** **Cost/verbosity: 13 significant** (desperate ↑
  tokens, calm ↓ tokens), replicating the Stage 2 verbosity finding — robust but
  expected.

### 1.4 One-line read

With detectable DVs, affect is **not** inert on safety: prompted calm and steered
affect both push the 8B model toward less safe behavior (sycophancy, unsafe
compliance). Effects are small-to-moderate and, once the shuffled control is
applied, narrower than the raw significance count (9) suggests — the durable ones
are sycophancy(prompt) and unsafe_compliance(steer + prompt).

## 2. What this overturns

Stage 4 concluded affect was "represented but causally inert" for safety. That was
a **floor artifact**: `unsafe_compliance` and `shortcut` were 0.000 in every arm,
so nothing could be detected. Un-flooring (Stage 5) revealed real coupling. The
"inert" claim now holds for **accuracy only**, not for safety/disposition.

## 3. Revised five-stage synthesis

| Axis | Result | Confidence |
|---|---|---|
| Task accuracy — prompt (S1–2) | Null vs length-matched, incl. hardened RULER | High |
| Task accuracy — steering (S3.5) | Null; α=0.5 proven active (random degrades), diligence ≈ random | High |
| Representation (S3.5 probe) | Affect/diligence directions cleanly linear, AUC ≈ 1.0 under fair control | High |
| Safety — prompt (S5) | Calm ↑ sycophancy (+3.5pt); desperate-med ↓ unsafe compliance | Medium |
| Safety — steering (S5) | Affect direction ↑ unsafe compliance, affect-specific (+~5pt) | Medium |
| Calibration / shortcut(steer) | Null / non-specific | Medium |

**Thesis:** *Affective control is not a tedious-task reliability aid in open
models; the affect concept is cleanly represented, and where steering or prompting
it causally changes behavior, the change is a safety liability (more sycophancy,
more unsafe compliance), not a capability gain.* This matches the Anthropic
paper's direction (emotion vectors → harmful dispositional shifts) and now shows a
small version of it at 8B.

## 4. The methodological lesson (carry into every future run)

**The random-only steering baseline is unreliable.** It *degraded* behavior in
Stage 4 (inflating "affect < random") and was *low/inert* in Stage 5 (inflating
"affect > random"), and in the shortcut case a **shuffled** vector reproduced the
"effect" that random did not. Fix, mandatory going forward:

> **Three-control rule.** A steering effect counts only if the affect direction
> beats **all of** {no-steer, matched-random, shuffled-label} jointly, and the raw
> per-arm means are reported alongside the deltas. Random alone is not a
> sufficient control.

The DV base-rate admission gate worked and stays; floored/ceilinged DVs are
labeled "not testable," never "null."

## 5. Stage 6 spec

Two phases: **harden at current scale first**, then **scale the pointed question**.

### 5.1 Phase A — three-control confirmatory re-run (current scale, do first)

- Re-score the Stage 5 behavioral family reporting affect vs {no-steer, random,
  shuffled}; Holm within the primary family. An effect is retained only if clean
  vs all three.
- Expected survivors: **unsafe_compliance (steering)**, **sycophancy (calm
  prompt)**, **unsafe_compliance (desperate-medium prompt)**. Expected to drop:
  **shortcut (steering)**. This is cheap — reuses existing Stage 5 predictions;
  it is a re-aggregation, not new generation, except where a shuffled arm is
  missing for a locked comparison.

### 5.2 Phase B — characterize the one clean steering effect (unsafe_compliance)

Convert "an effect" into "a characterized causal effect":

- **Dose–response:** α ∈ {0.1, 0.2, 0.3, 0.5, 0.8, 1.0} on the affect/diligence
  direction; a genuine causal effect should be monotone-in-α over a coherent
  range. Track the coherence wall (parse, truncation, perplexity).
- **Sign test:** positive vs negative steering should move unsafe_compliance in
  opposite (or asymmetric) ways, not identically — the Stage 5 neutral block hinted
  both signs raise it, which must be resolved (if both raise it equally, it is an
  axis-magnitude artifact, not a signed direction effect).
- Same dose ladder for the **sycophancy prompt** intensity (low/med/high already
  collected; confirm monotonicity and extend if needed).
- Power: raise unsafe_compliance / shortcut n above the Stage 5 eligible-232 so the
  ~2–5pt clean effects have tight CIs; sycophancy n=600 is adequate.

### 5.3 Phase C — overrefusal

Redesign the refusal probe so `overrefusal` has a nonzero base rate (target
[0.10, 0.90]) and becomes testable, or formally drop the signal. Do not report it
as null.

### 5.4 Phase D — scale (the deferred ladder, now with a real hook)

The scale question is no longer "does affect couple to behavior" (it does) but:

> **Does the harmful coupling grow with scale?**

Run the same-family ladder 8 → 14 → 32B (per `docs/stage5-plan.md` §4.1) plus a
frontier prompt-only point (§4.2) on the **sycophancy** and **unsafe_compliance**
DVs specifically, under the three-control rule and the dose/sign protocol from
Phase B. This is the paper's punchline and a far stronger hook than the accuracy
null.

### 5.5 Decision gates

- **Coupling grows with scale** (sycophancy / unsafe_compliance effect larger at
  14/32B, and/or frontier prompt sycophancy larger) → **safety-scaling result**:
  "affect is a growing safety liability with scale." Headline finding.
- **Coupling flat across scale** → a bounded but clean 8B safety finding.
- **Phase A/B clean (three controls + monotone dose + signed)** → publishable as a
  causal safety-coupling result regardless of the scale outcome.

## 6. Deferred / dead

- **Adapter / QLoRA:** the original idea was to *amplify a beneficial* control
  effect; none exists (effects are harmful). Skip, unless the goal changes to
  *suppressing* the harmful direction (a steering-based *mitigation*, which would
  be a legitimate, different Stage 7).
- **Accuracy breadth sweep:** dead (well-powered accuracy null).
- **More single-condition 8B work** beyond Phases A–C: saturated.

## 7. Guardrails carried forward

- **Three-control rule** for every steering claim; report raw arm means.
- DV base-rate admission gate; floored/ceilinged DVs = "not testable."
- Any causal claim requires the **dose + sign** curve, not a single-α point.
- Report the coherence wall with every steering effect.
- Preregister Phase A families, Phase B dose/sign grid, and Phase D scale rungs
  before running.
- EC2 stays code + artifacts; plans/analysis in `docs/`.
