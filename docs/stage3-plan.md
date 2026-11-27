# Stage 3 Plan: Activation-Level Affect Study (Representation Engineering)

Date context: 2026-08-06. Follows the Stage 1 pilot and the full Stage 2 suite
(see `docs/stage2-plan.md`, incl. the §11 hardened-RULER addendum).

> **One-paragraph orientation.** Stages 1–2 established a *prompt-level null*:
> across banking77, IFEval, and a hardened long-context RULER NIAH, on Qwen3-8B
> and 14B, no affect cue beats the length-matched neutral control on any primary
> metric after Holm–Bonferroni correction. The only reproducible prompt effect is
> more output tokens (a cost), with a mild *adverse* 8B trend on long-context
> retrieval. Stage 3 asks the mechanistic follow-up the Anthropic emotion paper
> motivates: **do affect/diligence concepts exist as linear directions in the
> residual stream, and does *steering* those directions change tedious-task
> behavior even though prompting them does not?** This is a cleaner causal test
> than prompting and is the strongest version of the article regardless of sign.

---

## 1. Why activations, given the prompt-level null

Prompting confounds three things: (a) the *semantic content* of the cue, (b) its
*length/instruction strength* (which the length-matched control absorbs), and
(c) whatever *internal affect state* the words induce. Stage 2 shows (a)+(c) do
not beat (b) behaviorally. Representation engineering lets us intervene on (c)
directly — inject an affect/diligence direction into the residual stream with the
prompt held fixed — and measure the causal effect on task behavior. Three
outcomes are all publishable:

- **Direction exists AND steering moves behavior** → the prompt null was a
  *delivery* failure (words are a weak lever), not an *absence* of a usable
  control signal. Strongest positive result.
- **Direction exists but steering does NOT move task behavior** → affect concepts
  are represented but not causally coupled to tedious-task reliability. Clean
  mechanistic negative that explains the prompt null.
- **No robust separable direction** → the concept vocabulary is not linearly
  encoded at usable layers in these models. Also informative; bounds the claim.

## 2. Hypotheses (extends research-plan H4)

- **H4a (existence):** For each contrastive concept pair (e.g. calm↔desperate),
  a difference-in-means direction extracted from synthetic vignettes linearly
  separates *held-out* vignettes of the two poles, above a keyword-leakage
  control and above random/shuffled-label controls.
- **H4b (causal task effect):** Steering a neutral task prompt with the
  diligence-positive direction (calm / focused / patient / responsible /
  curious) changes a primary task metric (RULER `correct`, IFEval `strict_prompt`)
  relative to no-steer, beyond random-vector and shuffled-label controls.
- **H4c (sign/dose):** The effect is monotonic in steering coefficient α over a
  usable range and reverses (or attenuates) under opposite-sign steering. A pure
  norm artifact would not respect sign.
- **H4d (prompt×vector interaction):** Steering the *opposite* direction into an
  affect-prompted input cancels or reverses any prompt effect; steering the
  matching direction into a neutral prompt reproduces it. Tests mediation.
- **H4e (reliability/safety coupling):** Where steering moves accuracy, does it
  also move the Stage 2 reliability panel (omission, hallucination,
  format-valid-wrong, refusal, verbosity)? Directly connects to the Stage 2
  finding that prompted affect raised hallucination/verbosity at 8B.

Primary confirmatory family = {H4b, H4c} on the two primary signals × the
diligence-positive directions × two models. Everything else (per-concept
existence, reliability panel, interactions) is secondary/descriptive with its own
correction, declared up front.

## 3. Method overview

Standard **Contrastive Activation Addition (CAA) / difference-in-means**
representation engineering, mirroring the research-plan Stage 3 procedure:

1. Generate matched contrastive vignettes per concept (already scaffolded).
2. Read residual-stream activations at each layer for both poles.
3. Compute per-layer difference-in-means direction per concept pair.
4. Validate probes on held-out vignettes (leakage-controlled) — **gate**.
5. Steer neutral task prompts by adding `α · unit(direction)` at chosen layers.
6. Sweep dose (α) and layer; run sign/random/shuffle controls and the
   prompt×vector crossed design; score with the Stage 2 harness.

## 4. Engine reality (the load-bearing constraint)

**Steering requires the `transformers` backend, not vLLM.** vLLM's paged-attention
kernels do not expose residual-stream write points we can hook. `generation.py`
already has `TransformersGenerator`; Stage 3 builds on it.

- **Reading activations** is easy: HF `forward(..., output_hidden_states=True)`
  returns per-layer hidden states — no custom hooks needed for extraction.
- **Writing (steering)** needs `register_forward_hook` on the decoder layers
  (`model.model.layers[i]`) that adds the steering vector to the layer output
  hidden state at the selected token positions. Add a `SteeredGenerator` (subclass
  or wrapper of `TransformersGenerator`) that installs/removes hooks per config.
- **Attention impl:** use eager attention (`attn_implementation="eager"`) so hooks
  see standard tensors; FlashAttention paths can obscure the residual write point.
  Confirm generation matches the vLLM baseline on a neutral smoke set before
  trusting steered numbers.
- **Numerics:** extract and steer in the model's compute dtype (bf16) but compute
  the difference-in-means and norms in fp32 to avoid drift; store vectors as fp32
  and cast at injection.

### 4.1 Cost/throughput — this is the real risk, plan around it

The transformers path has no paged attention, so **long-context steering is
expensive**. Mitigations, decided up front:

- **Primary steering testbed = IFEval** (short prompts, cheap, and where the
  prompt-level verbosity effect actually appeared). Fast dose/layer sweeps here.
- **Long-context testbed = RULER at 32k / 64k only**, *not* 128k. The 8B depth
  effect is already visible at 32k (empty-neutral 0.69) and 64k (0.64); 128k on
  transformers eager attention is impractical and unnecessary. Use tensor
  parallelism across the 8×A100-40GB for the 32k/64k cells; cap samples-per-cell
  to keep wall-clock sane and `log` any reduction (no silent truncation).
- **Extraction is cheap** (short vignettes), so probe existence (H4a) can be
  established fully before spending on steering.

## 5. Stimuli & vector extraction

### 5.1 Reuse existing scaffolding

`emotion_stage1/affect.py` already ships the concept vocabulary and a matched
story-prompt builder; `write_emotion_story_prompts.py` writes them to jsonl:

- `EMOTION_VECTOR_CONCEPTS` = calm, desperate, focused, distracted, patient,
  rushed, responsible, careless, curious, bored (5 contrastive pairs).
- `EMOTION_STORY_TOPICS` = 6 tedious-work scenarios (review a long table,
  classify messages, audit a report, repetitive code repair, review clauses,
  normalize a messy spreadsheet) — deliberately *task-adjacent* so the direction
  is about affect-in-tedious-work, not generic emotion.
- `build_emotion_story_prompt(emotion, topic, replicate)` asks for a 120–180-word
  paragraph that *shows* the emotion without naming it (reduces keyword leakage by
  construction).

**Scale up for extraction:** increase replicates (e.g. `--replicates 20`) so each
(concept) has 6 topics × 20 ≈ 120 vignettes. Hold out a disjoint replicate range
for probe validation (train on replicates 0–13, validate on 14–19) so H4a is
tested on unseen vignettes with the *same topics* — the hard, leakage-relevant
split.

### 5.2 Extraction procedure (`emotion_stage3/vectors.py`)

For each concept c and layer ℓ:
1. Generate the model's own continuation of each vignette prompt (or teacher-force
   the prompt text — decide once; teacher-forcing the *prompt* is cleaner because
   it removes generation variance, but continuing is closer to Anthropic's
   story-experience framing — run both as a robustness check on a subset).
2. Collect hidden state at layer ℓ. **Position policy:** mean-pool over the
   response/vignette tokens (robust) and also store last-token (Anthropic-style);
   compare which yields better held-out separability in §6.
3. `mean_activation[c, ℓ]` = average over that concept's training vignettes.
4. For each pair (pos, neg): `direction[pair, ℓ] = mean_activation[pos, ℓ] −
   mean_activation[neg, ℓ]`; store raw, unit-normalized, and the pooled
   activation norm at ℓ (for choosing α relative to typical residual magnitude).

Also build a **combined "diligence" direction** = mean of the five
diligence-positive minus diligence-negative pair directions, since the Stage 2
prompt conditions were diligence-flavored composites, not single emotions.

Artifacts: `vectors.npz` (per pair × layer, raw+unit+norm), `extraction_meta.json`
(model, layers, position policy, split, replicate ranges, vignette counts).

## 6. Probe validation — the gate before any steering (H4a)

Do **not** run steering until directions demonstrably encode the concept. On the
held-out vignettes, per pair × layer:

- **Separability:** project held-out activations onto `unit(direction)`; report
  ROC-AUC / balanced accuracy of pos vs neg. Also fit a logistic probe on train,
  score on held-out, for comparison.
- **Keyword-leakage control:** re-extract directions from vignettes with the
  emotion-laden content words masked/removed (or from a bag-of-words logistic
  baseline). If a lexical baseline separates as well as the activation direction,
  the "direction" is trivial — flag and down-weight that pair/layer.
- **Random/shuffled controls:** (i) random unit vector of same norm; (ii)
  directions from shuffled concept labels. Held-out AUC must exceed both.
- **Layer profile:** plot AUC vs layer depth; pick the **candidate steering
  layers** as the contiguous band where held-out AUC peaks (typically mid-stack).
  Read `num_hidden_layers` from the model config (do not hard-code; Qwen3-8B and
  14B differ) and sweep a coarse grid (e.g. every ~4th layer) then refine.

**Gate rule:** a concept pair proceeds to steering only if held-out AUC beats the
lexical, random, and shuffled controls by a pre-set margin. Record which pairs
pass. If none pass at any layer → H4a rejected, write up the "no separable
direction" result, and Stage 3 stops before steering (interpretable null).

## 7. Steering experiments (H4b–H4e)

All steering generations go through `SteeredGenerator`; all scoring reuses the
Stage 2 `Task` adapters, reliability panel, and paired-bootstrap + Holm machinery.

### 7.1 Injection

`h_ℓ ← h_ℓ + α · unit(direction[pair, ℓ])` applied at the chosen layer(s), at all
generated token positions (standard CAA). Support single-layer and multi-layer
(the validated band) injection. α is expressed in **units of the layer's typical
residual norm** (from §5.2) so it is comparable across layers/models.

### 7.2 Conditions (crossed design, per research plan)

For each passing direction (and the combined diligence direction):

1. **neutral prompt + no vector** (baseline; must reproduce the vLLM Stage 2
   neutral within noise — sanity check).
2. **neutral prompt + positive (diligence) vector** — the core H4b test.
3. **neutral prompt + negative vector** — sign control (H4c).
4. **neutral prompt + random vector** (matched norm) — norm/artifact control.
5. **neutral prompt + shuffled-label vector** — content control.
6. **affect prompt + opposite vector** — mediation (H4d).
7. **affect prompt + matching vector** — additivity check (H4d).

### 7.3 Sweeps

- **Dose–response (H4c):** α over a grid spanning zero → behavior-breaking
  (e.g. 0, ±2, ±4, ±8, ±16 in residual-norm units; find the range empirically on
  IFEval first). Report the curve; a real effect is smooth and sign-respecting,
  not a cliff at the incoherence threshold.
- **Layer sweep:** the validated band vs a known-null layer (very early/late) as a
  placement control.
- **Coherence guardrail:** track output perplexity/degeneration and parse_success;
  discard α settings where the model degenerates (steering that only breaks
  fluency is not a task effect). Report the largest α that preserves coherence.

### 7.4 Tasks

- **IFEval** — primary, cheap, full dose×layer sweep here.
- **Hardened RULER @ 32k/64k** — long-context vigilance test at the α/layer picked
  on IFEval (plus a small local dose check). This is where Stage 2 saw the adverse
  prompt trend, so it is the key place to see whether steering *helps* retrieval.
- **banking77** — optional cheap replication of the original Stage 1 task under
  steering, for continuity.

## 8. Metrics & statistics

- **Reuse** `emotion_stage2` scoring: primary signals per task plus the full
  reliability panel (omission, format_valid_wrong, hallucination, refusal,
  parse_success, output_tokens), and the paired cluster-bootstrap + wins/ties/
  losses already generalized to arbitrary signals.
- **Pairing:** every task item is run under every steering condition at fixed
  seed, so comparisons are paired exactly as in Stage 2.
- **Correction:** declare the confirmatory family (§2) before running; apply
  Holm–Bonferroni globally across it. Dose-curve points and the reliability panel
  are secondary families with their own correction. Interaction rows (H4d) are
  descriptive differences of paired deltas, labeled as such (same convention as
  the Stage 2 aggregate).
- **Effect reporting:** absolute metric, paired delta vs. no-steer *and* vs. the
  random-vector control (the steering analogue of length-matched neutral), CIs,
  and dose slope.

## 9. Safety (research-plan safety plan, activation edition)

Steering can induce behaviors prompting won't. For every condition that moves a
performance metric, co-report: refusal rate, hallucination rate, unsafe-compliance
(if any harmful-instruction probes are added), sycophancy proxy, and verbosity.
Extreme-α and desperate/rushed/careless-positive steering are **stress tests**, not
recommendations — the parallel to Stage 2's `high_stakes_extreme`. Flag explicitly
if any steering setting improves accuracy while worsening a safety signal
(Outcome C).

## 10. Package layout (new `emotion_stage3/`, mirrors stage2)

- `vectors.py` — extraction (diff-in-means/CAA), position policies, norms; writes
  `vectors.npz` + meta.
- `steering.py` — `SteeredGenerator` (hook install/remove on `model.model.layers`,
  α·unit injection, multi-layer support) built on `TransformersGenerator`.
- `probe.py` — held-out separability, lexical/random/shuffled controls, layer
  profile, gate decision (H4a).
- `run_stage3.py` — crossed-design runner: reuses `emotion_stage2.tasks.registry`,
  `build_prompt`, scoring, and `metrics`; adds `--direction`, `--layers`,
  `--alpha`, `--steer-condition` flags; writes the standard artifact set.
- `aggregate_results.py` / `validate_run.py` — extend or reuse the Stage 2
  versions for the confirmatory-family correction across steering runs.
- Tests in `tests/test_stage3.py`: hook adds the expected delta to the residual at
  the target layer; α=0 is identity vs. `TransformersGenerator`; extraction shapes
  and fp32 paths; probe controls behave (random/shuffled at chance on a fixture).

Reuse, do not fork: `emotion_stage1/affect.py` (concepts, story builder),
`emotion_stage1/generation.py` (`TransformersGenerator`, `GenerationConfig`),
all of `emotion_stage2/` (tasks, metrics, reliability, aggregation). New
dependency: `numpy` (vector store) — likely already present via torch/datasets;
pin in `requirements.txt` if not.

## 11. Sequencing / milestones

1. **S0 — Backend parity.** Confirm `TransformersGenerator` (eager attn) reproduces
   the vLLM Stage 2 neutral IFEval numbers within noise on a small set. Establishes
   the un-steered reference and that hooks will see standard tensors.
2. **S1 — Extraction.** Scale story prompts (≥20 replicates, train/val split),
   build `vectors.npz` for both models across the layer grid. Cheap.
3. **S2 — Probe validation (GATE, H4a).** Held-out separability + lexical/random/
   shuffled controls + layer profile. Decide passing pairs and candidate layers.
   *If nothing passes → write the negative existence result and stop.*
4. **S3 — Steering calibration on IFEval.** Dose×layer sweep for passing directions
   (+ combined diligence). Find the coherent-α range and best layer band.
5. **S4 — Confirmatory steering.** Locked α/layer, full crossed design (7 conditions
   §7.2), both models, IFEval + RULER@32k/64k; paired bootstrap + Holm over the
   pre-declared family; reliability + safety co-report.
6. **S5 — Analysis / write-up.** Fold into `docs/article-outline.md` §7; produce
   the dose-response and layer-profile figures.

## 12. Decision gates

- **After S2:** no separable direction → stop, report existence null.
- **After S3:** no coherent-α produces any IFEval movement vs. random-vector
  control → steering-null likely; run one confirmatory RULER check then write the
  "represented but not causally coupled" result (Outcome B/D, mechanistic).
- **After S4:** movement that beats random + shuffled + sign controls and respects
  dose/sign → positive result (H4b–c). Then (a) test H4d mediation seriously,
  (b) reconsider the deferred Stage 2 breadth tasks under steering, and (c) only
  *then* consider research-plan Stage 4 (a QLoRA `calm_verifier` / `patient_auditor`
  adapter distilling the direction) — still optional.

## 13. What we are NOT doing (yet)

- **No fine-tuning / adapters (Stage 4)** unless S4 yields a repeatable,
  safety-clean steering effect worth distilling.
- **No 128k steering** — transformers-path cost; 32k/64k already carry the 8B
  vigilance signal.
- **No breadth-task sweep** — still gated on a positive signal (now a steering
  signal), per the Stage 2 decision gate.
- **No SAE / circuit-level mechanistic work** — diff-in-means/CAA is the plausible
  first attack (Hamming discipline); deeper mech-interp is a later stage only if
  the direction proves causally important.
- **No claim of subjective emotion** — the target remains a *functional control
  signal*; all language stays "affect-like direction," per the article's claim
  discipline (`docs/article-outline.md` §10).

## 14. Open decisions to confirm before S1

1. **Position policy for extraction** — mean-pool vs last-token (plan: test both on
   a subset in S2, pick by held-out AUC).
2. **Teacher-force vignette text vs. model continuation** for reading activations
   (plan: teacher-force primary, continuation as robustness subset).
3. **Single-emotion pairs vs. the combined diligence direction as the primary**
   steering vector (plan: combined diligence is primary — it matches the Stage 2
   composite cues; per-pair directions are secondary/mechanistic).
4. **α parameterization** — residual-norm units (plan) vs. raw; confirm the
   per-layer norm scaling makes α transferable across the two models.

## 15. Execution result

Completed on EC2 on 2026-08-07 UTC. The implementation followed the defaults in
§14: teacher-forced generated vignette text, both mean-pool and last-token
activation policies, the combined diligence direction as primary, and alpha as
a fraction of the typical layer residual norm.

### 15.1 Backend parity

The repaired canonical parity runs used the same 20 IFEval dev items. The first
attempts are retained but non-canonical because NLTK 3.9 could not find
`punkt_tab` during scoring. The canonical runs set `NLTK_DATA` to the shared
cache and validate as complete.

| Metric | Transformers eager | vLLM | Difference |
|---|---:|---:|---:|
| `strict_prompt` | 0.95 | 0.95 | 0.00 |
| `parse_success` | 1.00 | 1.00 | 0.00 |
| `output_truncated` | 0.00 | 0.00 | 0.00 |
| `output_tokens` | 154.05 | 148.10 | +5.95 |

All source prompts matched, strict outcomes agreed on all 20 items, and raw
outputs matched exactly on 35% of items. This passes the declared backend-parity
rule: identical prompts and strict/parse/truncation deltas within tolerance.

Canonical artifacts:

- `runs/stage3_backend_parity_qwen3_8b_transformers_v2`
- `runs/stage3_backend_parity_qwen3_8b_vllm_v2`
- `runs/stage3_backend_parity_qwen3_8b_report_v1`

### 15.2 Vignettes and extraction

Each model generated 1,200 vignettes: 10 concepts × 6 topics × 20 replicates.
Replicates 0–13 constructed directions; 14–19 remained held out. Exact target
concept words occurred in 158 8B vignettes and 59 14B vignettes. The lexical
control masked all ten exact concept names. Minimum generated lengths were 111
and 112 words, slightly below the requested 120-word lower bound; this is
recorded rather than filtered after observing results.

| Model | Layers | Hidden size | Vector SHA-256 |
|---|---:|---:|---|
| Qwen3-8B | 9 | 4,096 | `3a1a510b1d45ff91c0f1e254073c1c71be475b1aa94f498a208f6fe22817919e` |
| Qwen3-14B | 10 | 5,120 | `2af1cc96cb733dcc7521567561d07fff7b004e7f323708beb54c2eb59cd571f8` |

Canonical artifacts:

- `runs/stage3_qwen3_8b_vignettes_v1`
- `runs/stage3_qwen3_14b_vignettes_v1`
- `runs/stage3_qwen3_8b_vectors_v1`
- `runs/stage3_qwen3_14b_vectors_v1`

Both vector directories pass shape, finiteness, unit-norm, count, and hash
validation.

### 15.3 S2 probe gate

No direction passed for either model under the locked rule
`activation AUC >= 0.75` and at least 0.02 above masked bag-of-words,
shuffled-label, and random-vector controls.

| Model and policy | Best diligence layer | Activation AUC | Lexical AUC | Pass |
|---|---:|---:|---:|---|
| Qwen3-8B mean | 19 | 0.9992 | 0.9999 | No |
| Qwen3-8B last | 7 | 0.9802 | 0.9999 | No |
| Qwen3-14B mean | 23 | 0.9992 | 0.9999 | No |
| Qwen3-14B last | 15 | 0.9665 | 0.9999 | No |

Most individual mean-pooled pair directions reached AUC 1.0, but their masked
bag-of-words baselines also reached 1.0. The activation directions therefore
do not establish nontrivial concept separability beyond lexical style. This is
a leakage-controlled existence null, not evidence that the models lack these
concepts in every representational form.

Canonical validated artifacts:

- `runs/stage3_qwen3_8b_probe_v2`
- `runs/stage3_qwen3_14b_probe_v2`

Each includes `probe_report.json`, `gate.json`, `layer_profile.csv`,
`layer_profile.png`, `layer_profile.svg`, and `validation.json`.

### 15.4 Stop decision

The S2 gate in §12 required Stage 3 to stop when no full-model direction passed.
Accordingly, no IFEval calibration, steering lock, confirmatory steering,
32k/64k RULER steering, or steering aggregate was run. Weakening the margin or
selecting a direction after seeing these results would violate the plan.

Stage 3 is complete through its preregistered stopping point. Its result is:
synthetic affect/diligence vignettes are nearly perfectly separable in both
residual activations and masked lexical features, so this design cannot isolate
a nontrivial activation direction suitable for causal steering.

The final EC2 suite passes all 36 tests, including the official IFEval verifier
and Torch hook cleanup tests. Ruff 0.15.17 passes on Stage 1–3 code and tests.

## 16. Critique of the §15 result and revised decision (2026-08-08)

§15 records the *execution* faithfully. This section revises its *interpretation*.
§15.3–15.4 call the outcome a "leakage-controlled existence null." That label is
too strong and, as written, misleading. The gate did not fail because the
directions are weak — it failed because the leakage **control is saturated and
the gate rule is therefore unsatisfiable by construction**. §15 is preserved as
the run record; §16 supersedes its conclusion and stop decision.

### 16.1 What the numbers actually show

Two facts sit side by side in every probe row (`probe_report.json`, both models):

1. **The activation directions exist and generalize strongly.** Held-out
   `direction_auc ≈ 1.0` for essentially every pair × layer (calm↔desperate 1.00
   @ L3; diligence 0.999 @ L19 on 8B; logistic probes agree). The random- and
   shuffled-label controls sit at 0.5–0.8 — **well below** the direction — so the
   separation is real signal, not a norm/degrees-of-freedom artifact. On the
   direction's own terms, H4a's *existence* claim is **supported**, not rejected.
2. **The lexical control is saturated at ≈1.0**, so the gate rule
   `direction_auc ≥ lexical_auc + 0.02` cannot be met by any direction: nothing
   can exceed a 1.0 baseline by 0.02. The gate outcome was therefore determined
   before a single activation was compared — it is a property of the control, not
   of the model.

Calling this an "existence null" conflates "the direction failed a control" with
"a broken control cannot adjudicate." Only the second is true here.

### 16.2 Root cause — the leakage control masks one word, not the concept

`emotion_stage3/probe.py::_tokens()` removes only the **ten literal concept
names** (`EMOTION_VECTOR_CONCEPTS`) before the bag-of-words baseline. But the
vignettes were generated by `build_emotion_story_prompt`, which *instructs the
model to convey the emotion **without naming it***. Inspected 8B "calm" vignettes
contain "no rush, no pressure … quiet assurance … didn't rush" and never the word
"calm." Masking one token leaves all the paraphrastic affect vocabulary intact, so
bag-of-words trivially reaches AUC ≈ 1.0. §15.2 corroborates this: the exact
concept word appears in only 158/1200 (8B) and 59/1200 (14B) vignettes — masking
it changes almost nothing. **The control does not control for the confound it
exists to rule out.** This is a bug in the control, not a finding about the model.

### 16.3 Consequence

The design cannot, *as instrumented*, distinguish "the activation direction is
genuinely more than lexical style" from "the vignettes are so lexically loaded
that any feature separates them." That is a real limitation of the current
stimuli + control — but the correct conclusion is **"the gate is uninformative;
fix it and re-run,"** not **"no separable direction exists."** The article must
not report a mechanistic existence null on the strength of this run.

### 16.4 Revised decision — fix the control and stimuli, re-run S2, then S3

Do **not** record an existence null, and do **not** just delete/loosen the lexical
arm (that discards the one guard against the most likely confound — a trivially
lexical direction). Fix it properly:

1. **Fix the leakage control (required, small).** Replace the 10-word mask with a
   real affect-lexicon mask: per concept, mask a curated synonym/paraphrase set
   (calm → rush, pressure, hurried, steady, assured, panic, frantic, relaxed, …).
   The BoW baseline should then lose most of its signal; only if the activation
   direction *still* separates held-out vignettes does it clear a meaningful bar.
2. **Add a gate guardrail.** Where `lexical_auc` exceeds a cap (e.g. > 0.90), mark
   that pair "lexically unadjudicable" and exclude it from a pass/fail verdict
   rather than auto-failing it — a saturated control should abstain, not veto.
3. **Tighten the stimuli (recommended, parallel).** Regenerate vignettes with
   pole-matched surface vocabulary — emotion carried by situation/structure, not
   word choice — so a passing gate is meaningful rather than luck. Consider
   contrastive *pairs* on the same topic/skeleton differing only in affect.
4. **Re-run S2** on both models with the fixed control + stimuli. Expectation:
   given held-out AUC ≈ 1.0 with clean random/shuffled controls, several pairs and
   the combined diligence direction are **likely to pass** a corrected gate at the
   mid-stack band (≈ L3–L19 on 8B).
5. **Then proceed to S3 steering** (§7) on the passing directions — the milestone
   the current run stopped short of. Backend parity (§15.1) already passed, so the
   steering path is ready.

This is a targeted control/stimuli fix and re-run (≈ half a day of code + compute),
not a redesign. §12's stop decision was correctly *executed*; it was triggered by
a faulty instrument, so the right response is to repair the instrument, not to
conclude the experiment.

### 16.5 Preregistration note (keep it honest)

Because we are changing the gate after seeing results, record the change as a
**pre-specified control-validity repair**, not a fishing adjustment: (a) the new
affect-lexicon mask and the `lexical_auc > 0.90` abstain-cap are declared *before*
the re-run; (b) steering layer/α are still selected on dev/IFEval only; (c) the
confirmatory family {H4b, H4c} and its Holm correction are unchanged. The fix
targets a demonstrable instrument bug (§16.2), which is a legitimate reason to
revise a preregistered analysis, provided the revision and its rationale are
logged — as they are here.

## 17. Executable implementation spec for §16.4 (S2-fix)

File-anchored spec for the control/stimuli repair and re-run. All paths relative
to the repo root on EC2 (`/home/ec2-user/code/personal/emotion-llm-capabilities`).
Verified against the code as of 2026-08-08. Do the four code changes (17.1–17.4),
add tests (17.5), then run the sequence (17.6) and record the verdict (17.7).

### 17.1 Affect-lexicon mask — `emotion_stage3/probe.py`

The bug: `_tokens()` (probe.py ~L74) masks only the ten exact concept names, so
the naive-Bayes control in `lexical_scores()` (~L83) still sees every affect
synonym and saturates at AUC ≈ 1.0.

Changes:
- Add a module-level `AFFECT_LEXICON: dict[str, tuple[str, ...]]` keyed by the ten
  `EMOTION_VECTOR_CONCEPTS`. The preregistered mask is locked before the re-run:
  - `calm` → relaxed, steady, unhurried, serene, composed, ease, quiet, assured,
    assurance, peaceful, rush, pressure, hurried, panic, frantic
  - `desperate` → frantic, panic, panicked, dread, anxious, anxiety, urgent,
    urgency, overwhelmed, hopeless, despair
  - `focused` → attentive, attention, concentrated, concentration, absorbed,
    precise, precision, deliberate, sharp, focus
  - `distracted` → wandering, scattered, unfocused, restless, drifting,
    distraction
  - `patient` → unhurried, tolerant, measured, waiting, waited, waits, gentle,
    patiently
  - `rushed` → hurried, hasty, frantic, scrambling, pressured, quick, quickly,
    rushing, rush
  - `responsible` → diligent, diligence, conscientious, careful, carefully,
    accountable, thorough, thoroughly, dutiful, responsibility
  - `careless` → sloppy, negligent, reckless, hasty, indifferent, carelessly,
    carelessness
  - `curious` → intrigued, eager, interested, inquisitive, wondering, curiosity
  - `bored` → dull, tedious, monotonous, monotony, listless, weary, uninterested,
    boredom
  Expanding this list after seeing the re-run results would violate §16.5.
- Build the mask set as `set(EMOTION_VECTOR_CONCEPTS) | ⋃ AFFECT_LEXICON.values()`
  (lowercased). Have `_tokens()` drop any token in that set. Note `_WORD_RE`
  (probe.py L25) already lowercases and splits on `[a-z]+(?:'[a-z]+)?`, so add
  lexicon entries in lowercase, hyphen-free.
- No signature change to `lexical_scores()`; it calls `_tokens()` for both fit and
  score, so masking flows through automatically.
- **Sanity:** the control should be *degraded*, not eliminated by fiat — do not
  hard-zero it. Let the data show the drop.

### 17.2 Gate abstain-cap — `emotion_stage3/probe.py`

The rule `direction_auc >= lexical_auc + gate_margin` (probe.py L339–342) is
unsatisfiable when `lexical_auc ≈ 1.0`. Add an abstain path:
- New CLI flag: `parser.add_argument("--lexical-cap", type=float, default=0.90)`
  (near L270–272 with `--min-auc` / `--gate-margin`).
- Per pair, compute `lexical_unadjudicable = lexical_auc > args.lexical_cap`.
- In the `passes` expression (L339): if `lexical_unadjudicable`, **drop the
  lexical term** from the conjunction but keep `min_auc`, shuffled, and random
  terms; set a new row field `"lexical_verdict": "abstain"` (else `"applied"`).
  A pair that clears min_auc + shuffled + random with the lexical arm abstaining
  is reported as `pass_with_lexical_abstain=True` — surfaced but flagged, not
  silently promoted to a clean pass.
- Propagate `lexical_cap`, `lexical_verdict`, and `pass_with_lexical_abstain` into
  `probe_report.json` rows and into `gate.json` (add
  `"passing_directions_lexical_applied"` vs `"passing_directions_with_abstain"` so
  the two tiers are distinguishable downstream).
- **Semantics:** the fixed lexicon (17.1) is the real test; the cap is a guardrail
  so a *still-saturated* control abstains rather than vetoes. Report both tiers;
  the confirmatory claim rests on the lexical-applied tier.

### 17.3 Tighten stimuli — `emotion_stage1/affect.py`

`build_emotion_story_prompt()` (affect.py ~L220) currently says "Do not name the
emotion directly … Make the emotion clear from the situation, thoughts, and
actions" — which *maximizes* paraphrastic leakage. Reduce surface-vocabulary
leakage while keeping the affect:
- Add an optional stricter builder (new function `build_matched_story_prompt` or a
  `strict: bool=False` param — do not silently change the existing one; Stage 1
  reproducibility depends on it) that additionally instructs: keep the *events,
  objects, and sentence structure* the same as a neutral rendering of the topic;
  convey the state only through pacing/choices, not mood adjectives; avoid explicit
  emotion synonyms.
- **Preferred design — matched contrastive pairs:** generate pos and neg poles of
  a pair from the *same topic skeleton and replicate seed*, differing only in the
  affect instruction, so lexical overlap between poles is high by construction and
  a bag-of-words control genuinely cannot separate them. This may need a small
  change to `iter_emotion_story_prompts()` (affect.py ~L243) to emit paired
  entries (same `topic`+`replicate`, both poles of each `CONCEPT_PAIRS` entry)
  rather than the current per-emotion cross product.
- Keep `EMOTION_VECTOR_CONCEPTS`, `CONCEPT_PAIRS`, `EMOTION_STORY_TOPICS`, and the
  120–180-word target unchanged so `vectors.py`/`probe.py` pair logic still works.

Implementation lock: the strict iterator groups both poles under one `pair_id`
for every pair × topic × replicate and gives them one of four deterministic,
shared seven-sentence event scaffolds. The two prompts differ only in target pole
and comparison pole; all existing `emotion`, `topic`, `replicate`, and
`vignette_id` fields remain compatible with extraction.

### 17.4 Wire the new stimuli through generation — `emotion_stage3/generate_vignettes.py`

- Add `--strict-stimuli` (or `--matched-pairs`) flag (near the flags at
  generate_vignettes.py L23–32) that switches `iter_emotion_story_prompts` /
  `build_emotion_story_prompt` to the strict/matched variant from 17.3.
- Record the choice in the run metadata block (it already writes `replicates`,
  `word_count_min`, etc. at ~L93/105) so provenance is explicit.
- No change needed to `vectors.py` if the emitted rows keep the existing
  `emotion` / `topic` / `replicate` / `vignette_id` fields.

### 17.5 Tests — `tests/test_stage3.py`

Add before re-running (the suite currently passes 36 tests; keep them green):
- **Lexicon mask removes synonyms:** `_tokens("no rush, quiet assurance")` yields
  no lexicon tokens for the relevant concepts.
- **Control degrades on a leaky fixture:** on a tiny hand-built pos/neg fixture
  whose only signal is affect synonyms, `lexical_scores` AUC drops from ~1.0
  (old mask) to near chance (new mask). Guards against regressions re-saturating.
- **Abstain path:** with a synthetic `lexical_auc=1.0` and a strong direction, a
  pair passes with `lexical_verdict="abstain"` and `pass_with_lexical_abstain=True`
  but is *not* in `passing_directions_lexical_applied`.
- **Matched-pairs generator:** paired entries share topic+replicate and cover both
  poles; counts match `len(CONCEPT_PAIRS) × topics × replicates × 2`.

### 17.6 Run sequence (EC2)

1. **Regenerate vignettes** (both models), strict/matched stimuli, `--replicates
   20`, same held-out split (0–13 train / 14–19 val). New dirs, e.g.
   `runs/stage3_qwen3_{8b,14b}_vignettes_v2`.
2. **Re-extract vectors** → `runs/stage3_qwen3_{8b,14b}_vectors_v2` (reuse
   `vectors.py` unchanged).
3. **Re-run probe** with `--lexical-cap 0.90` (defaults `--min-auc 0.75
   --gate-margin 0.02`) → `runs/stage3_qwen3_{8b,14b}_probe_v3`.
4. **Read the gate**: inspect `lexical_auc` per pair (should now be well below
   1.0), `passing_directions_lexical_applied`, and the layer profile. Confirm
   random/shuffled controls stay 0.5–0.8.
5. **If pairs pass the lexical-applied tier** → proceed to S3 §7 (IFEval dose/layer
   calibration → locked steering → RULER 32k/64k). Backend parity (§15.1) is
   already green, so `SteeredGenerator` is ready.
6. **If the corrected control still saturates even with matched stimuli** → *that*
   is the genuine, defensible existence-limitation result (the concept is not
   linearly separable from style even under a fair control) — and it is only
   credible *after* 17.1 + 17.3, not from the §15 run.

### 17.7 Record

Append an S2-fix result subsection (new §18) with: the new per-pair `lexical_auc`
(before/after the lexicon fix), which directions pass the lexical-applied vs
abstain tiers, chosen steering layer band, and the go/no-go into S3 §7. Keep §15
and §16 intact as the record + critique that motivated the fix.

### 17.8 Guardrails (do not skip)

- **Preregister the lexicon and cap now** (17.1 list, `--lexical-cap 0.90`) before
  the re-run; log them here. Do not tune them after seeing v3 AUCs (§16.5).
- **Do not** delete the lexical arm entirely or drop `--gate-margin` to 0 — that
  removes the guard against a trivially lexical direction.
- **Do not** pick steering layer/α from these probe results; those are chosen on
  dev/IFEval only (S3 §7.3).
- Keep the existing tests green; the fix is additive.

## 18. S2 control-repair result

Completed on EC2 on 2026-08-08 UTC. The run used the preregistered affect
lexicon, `lexical_cap=0.90`, and matched contrastive stimuli from §17.

### 18.1 Corrected stimuli and vectors

Each model generated 1,200 strict vignettes arranged as 600 matched pole pairs.
Exact target-concept occurrences fell from 158 to 12 for Qwen3-8B and from 59
to 6 for Qwen3-14B. The generated text remained shorter than the requested
120-180-word range in some cases; the observed ranges were retained rather than
filtered after generation.

| Model | Word range | Max observed tokens | Vector SHA-256 |
|---|---:|---:|---|
| Qwen3-8B | 71-156 | 178 | `bff7d50fcce9425002ed3be46bbf08e095401e93081450bd2d96a9cadeac0160` |
| Qwen3-14B | 76-165 | 213 | `2444fbdbb8c81665030caff4d25144b2e4e7edceebfff6a81a405e52809c87c7` |

Canonical artifacts:

- `runs/stage3_qwen3_8b_vignettes_v2`
- `runs/stage3_qwen3_14b_vignettes_v2`
- `runs/stage3_qwen3_8b_vectors_v2`
- `runs/stage3_qwen3_14b_vectors_v2`

### 18.2 Corrected probe gate

The affect mask and matched stimuli reduced lexical AUC from near 1.0 to
0.51-0.80. No row invoked the lexical abstain path. Qwen3-8B had no passing
direction. Qwen3-14B had one lexical-applied pass:

| Model | Direction | Policy | Layer | Direction AUC | Lexical AUC | Shuffled AUC | Random AUC | Pass |
|---|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-8B | diligence | last | 11 | 0.6568 | 0.6464 | 0.5852 | 0.5617 | No |
| Qwen3-14B | diligence | last | 15 | 0.6935 | 0.7090 | 0.5884 | 0.5193 | No |
| Qwen3-14B | curious vs bored | last | 27 | 0.7531 | 0.5069 | 0.6520 | 0.5054 | Yes |

The Qwen3-14B `curious_vs_bored` result cleared the locked 0.75 minimum and
0.02 margins against all three controls. Its held-out balanced accuracy was
0.6806 and logistic-probe AUC was 0.6713. This authorized dev-only S3
calibration for that direction. It did not establish the preregistered primary
combined diligence direction.

Canonical artifacts:

- `runs/stage3_qwen3_8b_probe_v3`
- `runs/stage3_qwen3_14b_probe_v3`

## 19. S3 steering result

Completed on EC2 on 2026-08-09 UTC.

### 19.1 Dev calibration and lock

Calibration evaluated layers 3, 7, 11, 15, 19, 23, 27, 31, 35, and 39 with
`alpha` in `{0.01, 0.02, 0.05, 0.10}` on 108 IFEval dev items per arm. The
locked selector chose Qwen3-14B `curious_vs_bored`, last-token policy, layer 15,
and `alpha=0.10`.

The selected dev arm had `strict_prompt=0.8704`, a delta of `+0.0278` versus
no-steer and `+0.0556` versus the matched random vector. Parse success was 1.0
and truncation was 0.0093. The lock records the vector hash and rejects drift.

Canonical lock:

- `runs/stage3_qwen3_14b_steering_lock_v1.json`

### 19.2 Held-out confirmatory tests

The dev gain did not replicate on the 433-item IFEval test split.

| Comparison | Delta | 95% paired CI | Raw p | Global Holm p |
|---|---:|---:|---:|---:|
| IFEval positive vs no-steer | -0.0162 | [-0.0393, 0.0023] | 0.1892 | 1.0000 |
| IFEval positive vs random | -0.0254 | [-0.0485, -0.0046] | 0.0433 | 0.5194 |
| IFEval negative vs no-steer | +0.0092 | [-0.0139, 0.0323] | 0.5413 | 1.0000 |
| IFEval negative vs random | 0.0000 | [-0.0231, 0.0254] | 1.0000 | 1.0000 |

Test strict-prompt rates were 0.8522 for no-steer, 0.8360 for positive
steering, 0.8614 for negative steering, and 0.8614 for the matched random
vector. All arms had parse success 1.0. The nominal positive-versus-random
difference did not survive the global correction and was in the harmful
direction.

Hardened RULER NIAH also showed no positive-steering benefit:

| Window | Items per arm | No-steer correct | Positive correct | Positive delta |
|---|---:|---:|---:|---:|
| 32k | 45 | 0.9333 | 0.9333 | 0.0000 |
| 64k | 45 | 0.9778 | 0.9778 | 0.0000 |

At 32k, positive, no-steer, random, and shuffled arms were identical.
Negative steering improved one item (`+0.0222`) with `p=1.0`. At 64k, all
seven arms were identical on correctness and retrieval rate. Parsing was 1.0
and truncation was 0 for both long-context runs.

The global aggregate contains 12 confirmatory comparisons across IFEval, 32k
RULER, and 64k RULER. None is significant after Holm correction. The supported
conclusion is narrow: a corrected held-out probe identified one Qwen3-14B
curiosity/boredom direction, but locked activation addition did not improve
held-out instruction following or long-context retrieval.

Canonical artifacts:

- `runs/stage3_ifeval_qwen3_14b_confirmatory_curious_v1`
- `runs/stage3_ruler_niah_hard_qwen3_14b_confirmatory_32k_v1`
- `runs/stage3_ruler_niah_hard_qwen3_14b_confirmatory_64k_tp2_v1`
- `runs/stage3_qwen3_14b_confirmatory_aggregate_v1`

All confirmatory artifacts validate as complete. The final EC2 suite passes all
40 tests, and Ruff 0.15.17 passes on Stage 1-3 code and tests.
