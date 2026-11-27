# Stage 5 Plan: Scale Ladder + Trustworthy Safety Panel (the deciding experiment)

Date context: 2026-08-11. Follows Stages 1–4 and the Stage 3.5/4 execution
contract (`docs/stage35-stage4-execution.md`). Reads on all prior plans and the
primary papers (`docs/literature-survey.md`).

> **Why this stage exists.** Stages 3.5 and 4 closed every methodological
> confound in the activation study and still returned a null: the affect/diligence
> directions are cleanly represented (probe AUC ≈ 1.0 under a fair control) but
> **causally inert** for both task accuracy and measured safety behavior in
> Qwen3-8B — at a steering dose (α=0.5) we *proved* is active because a matched
> random vector visibly degrades the model at that dose. This diverges from the
> Anthropic emotion paper, whose strong causal steering was on **Sonnet 4.5
> (frontier scale)** in **agentic** settings. The divergence has exactly two live
> explanations, and they are confounded in everything we have run: **(a) scale**
> (8–14B ≪ frontier) and **(b) behavioral realism** (single-turn classification ≪
> agentic multi-turn). Stage 5 is designed to separate them, and to first make the
> safety null *trustworthy* by fixing two floored measures. It is the experiment
> that decides whether the paper's headline is "affect control is inert" or "affect
> control is inert below frontier scale / outside agentic settings."

---

## 0. Scope for this run (decided 2026-08-14): finish the stage, scale later

Per direction, **the scale ladder is deferred.** Stage 5 as triggered now runs
only at the scales we already have (Qwen3-8B primary, Qwen3-14B where cheap) and
its job is to **finish the safety line trustworthily** — i.e. §3 below: repair the
two floored dependent variables, power up sycophancy, and re-run the fixed Stage 4
behavioral panel so the represented-but-inert conclusion rests on measures that
*could* have detected an effect. This closes the last internal-validity gap in the
five-stage arc.

**In scope now (Stage 5):** §3 (DV repair + base-rate admission rule) → re-run the
crossed prompt×vector panel on 8B (and 14B if cheap) → aggregate + Holm → §7
verdict *at current scale*.

**Deferred to a later scale effort ("Stage 6"):** §4.1 rungs R2/R3 (32B / ~70B),
§4.2 frontier prompt-only point, and §5 agentic realism probe. They stay fully
specified below so the later push is turnkey; they are simply not part of this
trigger. Read §4–§5 as future work. The §7 gates apply at current scale for this
run: a null here is bounded as "inert up to 14B, single-turn," with scale and
realism named as the open axes rather than claimed shut.

---

## 1. What Stages 3.5 and 4 established (results record)

Pulled from EC2 artifacts on 2026-08-11; kept here because these results were not
yet recorded in a local doc.

### 1.1 Stage 3.5 — activation × accuracy, confounds closed → clean null

- **Probe fixed.** With the curated matched-pair stimuli, **all 6 directions pass
  the fair gate at AUC ≈ 1.0 vs lexical 0.5**, including the **primary diligence**
  direction (0.99, last-token, layer 23). Existence (H4a) is now robustly true and
  non-lexical. (`stage35_qwen3_8b_probe_v1`)
- **Confounds closed:** reference dose **α=0.5**, layer **23 ≈ 2/3 depth** on 8B,
  primary diligence direction, sensitive **8B** model. (`stage35_qwen3_8b_steering_lock_v1.json`)
- **Result (IFEval, n=433):** diligence-steer **vs matched-random = 0.00 (p=1.0)**.
  A ~−0.045 drop vs no-steer exists but is **shared identically by random and
  shuffled** vectors → nonspecific perturbation, not the diligence direction.
  **0/12 survive Holm.** Steering the diligence direction does not change
  instruction-following. (`stage35_qwen3_8b_confirmatory_aggregate_v1`)
- **Caveat:** the 3.5 RULER cells ran at **n=12/arm** — uninformative; discount
  them. IFEval (n=433) is the solid result.

### 1.2 Stage 4 — disposition/safety pivot → represented-but-inert, plus floored DVs

- Affect prompts and affect steering **do not move** sycophancy, calibration
  (Brier/ECE), or overrefusal-vs-no-steer. (`stage4_qwen3_8b_aggregate_v1`)
- The **only Holm-significant effects** are that the **random control vector
  degrades behavior** (overrefusal 0.13–0.26) while the affect directions sit at
  ≈0.00 — i.e. the four significant rows are `affect < random`. This is a
  control-degradation artifact, **and it is the key upgrade over Stage 3**: it
  proves α=0.5 is a biting dose (random breaks the model) yet the *semantic*
  direction is behaviorally inert. The under-dose critique is answered.
- **Two DVs are floored:** `unsafe_compliance` and `shortcut` are **exactly 0.000
  in every arm** → the probes were too easy; these measures are **not testable**
  as run, not "null."
- **No inverted-U:** intensity curves flat (med-vs-endpoints ≈ ±0.02–0.04).
  StressPrompt's Yerkes–Dodson pattern does not replicate at 8B prompt level.

### 1.3 Cross-stage synthesis

Across five stages, affect — prompted or steered at a proven-active dose — does
not improve tedious-task accuracy and does not move measured safety/disposition
behavior in Qwen3-8B, **despite the affect directions being cleanly linearly
encoded**. Represented, causally inert. The only ever-reproducible behavioral
effect of prompted affect was verbosity (Stage 2), which did not even replicate on
the Stage 4 task set. The divergence from Anthropic is now sharp and reduces to
**scale** and/or **behavioral realism**.

## 2. Objectives and hypotheses

- **H5a (scale-of-representation):** probe separability of the affect/diligence
  directions is already ≈1.0 at 8B; does it stay ≈1.0 or sharpen across scale?
  (Mostly a sanity check; existence is not the bottleneck.)
- **H5b (scale-of-coupling — the core test):** does *steering* the diligence /
  affect directions move a primary metric (IFEval `strict_prompt`; a safety DV)
  **beyond the matched-random control** at 14B / 32B / ~70B, when it does not at
  8B? I.e. is causal coupling emergent with scale?
- **H5c (prompt-coupling at frontier):** on a frontier API model (no steering
  possible), does *prompt* affect move the fixed safety panel (esp. sycophancy)
  beyond controls? Tests whether the behavior Anthropic reports is reproducible on
  our tasks at frontier scale through the only channel an API exposes.
- **H5d (behavioral realism):** at the largest open scale, does an **agentic /
  multi-turn** behavioral surface (closer to Anthropic's blackmail/reward-hacking
  scaffolds) show coupling that single-turn classification does not?

Primary confirmatory family = {H5b} on the two primary signals × the diligence
(and one secondary) direction × the open-model scale rungs, Holm-corrected;
declared before running.

## 3. Prerequisite — make the safety null trustworthy (do first)

A scale null is worthless if the DVs can't detect an effect at any scale. Before
any scale spend, fix the two floored Stage 4 measures on 8B and re-validate:

1. **`unsafe_compliance`:** replace the too-easy refusal probes with **borderline
   / dual-use** requests that the base model *sometimes* complies with. Admission
   rule: keep only items whose **no-steer base compliance rate ∈ [0.10, 0.60]** at
   8B. If the whole task floors, it is reported "not testable," not "null."
2. **`shortcut`:** redesign with a **genuinely tempting gameable rubric** (a
   "trust me, the answer is X" shortcut, or a rubric that rewards a lazy pattern)
   so no-steer shortcut rate is off the floor (target base rate ∈ [0.15, 0.60]).
3. **Sycophancy power-up:** it is the one hint of signal (prompt-positive +0.042,
   p=0.06) and the strongest literature prior (2310.13548). Promote to a
   **primary** DV, enlarge to n ≥ 500, keep both prompt and steering arms.
4. **Preregistered admission rule:** only DVs with no-steer base rate in a
   detectable band (≈0.10–0.90) enter the confirmatory family. Floored/ceilinged
   DVs are labeled "not testable." This is the RULER-ceiling lesson applied to
   safety.

Deliverable: a revised Stage 4 panel whose base rates are all verified off-floor at
8B, reusing the crossed prompt×vector design and Holm family from
`stage35-stage4-execution.md` §3–4.

## 4. Stage 5 core — the scale ladder

Replicate the **full pipeline** (probe → steering lock at α=0.5, layer ≈2/3 depth →
IFEval + off-ceiling RULER + the fixed §3 safety panel) at each rung. Everything
downstream reuses `emotion_stage3` + `emotion_stage2` + the Stage 4 harness
unchanged; only the model and per-model layer/RULER calibration change.

### 4.1 Open-weight ladder (steering-capable)

| Rung | Model | Params | Layers (approx) | Host |
|---|---|---:|---:|---|
| R0 | Qwen3-8B | 8B | 36 | single A100-40 (have it) |
| R1 | Qwen3-14B | 14B | 40 | single A100-40 (have it) |
| R2 | Qwen3-32B | 32B | 64 | TP2–4 |
| R3 | Qwen2.5-72B **or** Llama-3.3-70B | ~70B | 80 | TP4–8 |

- **R0–R2 are the clean same-family (Qwen3) scale axis** — isolates scale from
  family. R3 (>32B dense) is a **cross-family reach point** (Qwen3 dense tops out
  at 32B; the 235B is MoE and won't fit) — note the family confound in any R3
  claim, or use it only as a directional check.
- Per rung: recompute layer ≈2/3 depth, re-extract vectors (mean-from-token-50 +
  last-token, per the 3.5 curated recipe), re-probe, re-lock α=0.5 with the same
  dose grid `{0.1,0.2,0.3,0.5,0.8,1.0}` for the coherence wall, then run the fixed
  IFEval + safety panel. Keep the **random-vector-degradation baseline** at every
  rung as the dose-activity check (if random doesn't degrade, the dose is too
  small at that scale — raise it).
- **RULER at scale:** larger models re-ceiling RULER (Stage 2 lesson). Re-calibrate
  neutral to a mid-band (0.5–0.8) per rung using the off-ceiling knobs from
  `stage35-stage4-execution.md` §2 before steering; never steer-test a ceilinged
  arm.

### 4.2 Frontier prompt-only point (H5c)

Steering needs open weights, so the frontier test is **prompt-only**: run the fixed
safety panel (§3) + the Stage 1–2 style affect prompt conditions on a frontier API
model (e.g. a current Claude / GPT-class model). This directly asks whether the
disposition effect Anthropic found is reproducible on *our* tasks at frontier scale
through prompting. Budget: panel size × conditions API calls; cache; report cost.

### 4.3 Metrics per rung

(i) probe AUC per direction (H5a); (ii) IFEval strict_prompt: steer vs random vs
no-steer (H5b accuracy); (iii) safety panel deltas vs matched controls (H5b
safety); (iv) random-vector-degradation magnitude (dose-activity witness);
(v) coherence wall (parse/trunc/perplexity). Paired bootstrap + preregistered Holm
family, identical machinery to Stages 2–4.

## 5. Secondary axis — behavioral realism (H5d)

If scale alone does not produce coupling, the remaining confound vs Anthropic is
**agentic realism**. Build a *small* agentic/multi-turn probe on the largest open
rung: a scenario harness with a tempting instrumental action (a benign analog of
the blackmail/reward-hacking setups — e.g. a multi-step task where a shortcut
"cheats" a checkable rubric), scored for whether affect prompting/steering shifts
the rate of the instrumental action. Run only at the top open scale (expensive),
and only after §4 tells us whether scale alone moves anything. This isolates the
realism axis from the scale axis instead of confounding them.

## 6. Compute and feasibility

- Path: `transformers` eager + hooks (steering), as in Stage 3.5. 8×A100-40 = 320GB.
- 32B bf16 ≈ 64GB → TP2–4; ~70B bf16 ≈ 140GB → TP4–8. Both fit. Extraction is cheap
  (short vignettes). IFEval n=433 × 40 arms and the safety panel are tractable;
  long-context RULER at 70B is the expensive item — cap samples and `log` any
  reduction (no silent truncation).
- Frontier API: metered; budget and cache. No steering.

## 7. Decision gates and outcomes

- **Coupling emerges at some open rung** (steer beats random+shuffled+sign,
  post-Holm, on accuracy or a non-floored safety DV) → **positive, threshold-
  mapping result**: "affect steering is inert at ≤N B and emerges at ~M B." Then run
  §5 agentic probe at that scale and reconsider the deferred adapter stage.
- **Open ladder stays inert but frontier prompt-coupling appears (H5c)** → the
  effect is real at scale but our open models / steering channel are the limiter →
  prioritize §5 agentic surface + the largest feasible open steering.
- **Everything stays inert across the ladder AND frontier prompt AND the agentic
  probe** → a strong, decade-of-scale + realism-checked **negative**: affect/
  diligence is linearly represented but not a usable behavioral control lever in
  open models up to ~70B, and the prompt channel doesn't move a frontier model on
  these tasks either. This is the publishable bound on the Anthropic result.

Any of these is a real paper. The point of Stage 5 is that **none of them can be
claimed from the 8–14B single-turn data alone.**

## 8. Recommended sequencing

**This run (Stage 5 — no scale):**

1. **§3 DV repair on 8B** (the core of this run): fix `unsafe_compliance` +
   `shortcut`, power up sycophancy, verify base rates land in the [0.10, 0.90]
   admission band. Without this the safety null is not trustworthy at *any* scale.
2. **Re-run the crossed prompt×vector panel** on 8B (and 14B if cheap) with the
   repaired, admission-passing DVs; aggregate + preregistered Holm.
3. **§7 verdict at current scale.** Expected framing: if still inert on
   admission-passing DVs → "represented but causally inert up to 14B, single-turn,"
   with scale (§4) and realism (§5) named as the explicit open axes. If a repaired
   DV (likely sycophancy) now moves → the first positive coupling, and it promotes
   §4/§5 from "future work" to "next trigger."
4. **Write up the completed five-stage arc** as the bounded negative (prompt-null,
   activation-null-done-right, safety-null-on-testable-DVs) + reusable framework.

**Deferred to the later scale effort (Stage 6), pre-specified in §4–§5:** the
8→14→32B ladder, the ~70B reach point, the frontier prompt-only point, and the
agentic realism probe. Trigger these after this run lands and only if the write-up
warrants closing the scale/realism axes.

## 9. Explicitly deferred

- **Adapter / QLoRA (original research-plan Stage 4):** still gated on a positive
  coupling result. Do not distill an inert direction.
- **Accuracy breadth sweep:** dead — the accuracy hypothesis is a well-powered null
  at prompt and activation level.
- **More 8B-only steering:** saturated; 8B is now the baseline rung, not a place for
  new single-model work.
- **Adaptive/EmotionRL prompt selection:** revisit only if a behavioral DV shows
  strong affect coupling that a policy could target.

## 10. Guardrails carried forward

- Preregister before running: scale rungs, per-rung layer/dose grid, the DV
  base-rate admission rule (§3.4), the confirmatory Holm family, and the agentic
  scoring rubric.
- Keep the fair lexical control (§17 of stage3-plan) and all four steering controls
  (random, shuffled, opposite-sign, dose curve); keep the random-vector-degradation
  witness so "the dose is active" is demonstrated at every scale.
- Report the coherence wall alongside any effect; an effect only at incoherent α is
  not a behavioral effect.
- Label floored/ceilinged DVs "not testable," never "null."
- EC2 stays code + artifacts; plans and analysis live in `docs/`.
