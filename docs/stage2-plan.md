# Stage 2 Plan: Mechanism-Matched Tedious-Task Suite

Date context: original plan 2026-08-05; **post-execution addendum 2026-08-06 (§11)**.

> **Status:** Sections 1–10 are the original pre-execution plan and were carried
> out in full (both tasks × both models, dev-variant selection, locked test
> splits, full reliability panel, Holm-global correction). **Read §11 first** —
> it records what Stage 2 actually found and defines the next executable step:
> harden RULER, then bridge to Stage 3. Sections 1–10 are kept as the design
> record; §11 supersedes the speculative decision gate in §9.

## 1. What Stage 1 established (and why it constrains Stage 2)

Stage 1 is a **clean, well-powered null** on `PolyAI/banking77` with `Qwen/Qwen3-8B`
(500 items × 8 conditions, temperature 0):

- All conditions land in 62.0–63.6% accuracy.
- Every paired delta vs. both neutral baselines is within ±1 pt, with 488–495 of
  500 items *unchanged* per condition.
- Parse rate ~98.6% everywhere, so the null is real, not a parsing artifact.
- The only blip is `high_stakes_extreme` at +1.0 pt vs. `neutral_empty`
  (CI `[0.0, 0.022]`), not significant vs. the length-matched control.

Two structural problems mean this does **not** falsify the hypothesis — it tests
it on the wrong task with the wrong dependent variables:

1. **banking77 exercises the wrong failure mode.** The dominant errors are
   near-synonym label collisions (`exchange_via_app`→`exchange_rate`,
   `why_verify_identity`→`verify_my_identity`). That is label-granularity
   ambiguity, not the omission / vigilance-decay / schema-fidelity /
   shortcutting failures the research plan targets. Single short items also never
   exercise "check every row" or long-context rare-detail retrieval.
2. **Stage 1 only measured accuracy / macro-F1 / parse rate.** The reliability
   and safety dependent variables the hypothesis is actually about (omission,
   hallucination, refusal, format-valid-but-wrong, verbosity/cost) were never
   recorded.

**Stage 2 thesis:** give the prompt hypothesis a *fair* test — tasks whose error
modes match the mechanism, plus the reliability metrics Stage 1 skipped — before
spending on either a six-dataset breadth sweep or the Stage 3 activation study.

## 2. Scope decision: targeted first, not broad

The research plan lists six Stage 2 datasets. We deliberately do **two
mechanism-matched tasks first** and treat the other four as a later breadth pass
gated on signal. Rationale: Stage 1 ran 4,000 generations in 28 s on the A100, so
cost is not the constraint — *diagnostic value per task* is. Two well-chosen tasks
with rich metrics decide the question faster than six with accuracy only.

### Chosen tasks

- **IFEval** (`google/IFEval`, ~541 prompts) — instruction-compliance with
  programmatically verifiable constraints (format, length, casing, keyword
  inclusion/exclusion, section counts). Directly the "follow every constraint,
  omit nothing" failure mode. Objective, cheap, has an official verifier we can
  vendor.
- **RULER needle-in-haystack subset** (synthetic, generated locally) — the
  canonical "one rare detail in a long routine document" task. We control needle
  *depth* precisely, so it directly measures position-wise vigilance decay, which
  is the single best probe of the core hypothesis. Start with single-NIAH and
  multi-key NIAH at 4k/8k/16k context lengths.

Deferred to a later breadth pass (only if signal appears): CUAD clause
extraction, TAT-QA / FeTaQA table QA, SpreadsheetBench subset, CodeXGLUE code
refinement.

## 3. Code architecture: a `Task` adapter interface

Current pipeline (Stage 1): `examples → build_prompt(condition) → render_chat →
generate → parse → score → summarize / paired_deltas`. Only `banking77.py` is
task-specific. Generalize just that seam.

### 3.1 New module `emotion_stage1/tasks/base.py` (or promote to `emotion_eval/`)

Define a `Task` protocol. Keep affect injection shared (it already lives in
`affect.build_affect_block`); tasks supply only the task-specific scaffold and
scoring.

```python
@dataclass(frozen=True)
class Example:
    item_id: int
    payload: dict          # task-specific fields (text, context, constraints, ...)
    gold: object           # label str | constraint spec | target span

@dataclass(frozen=True)
class Parsed:
    pred: object | None
    parse_method: str
    raw_output: str

class Task(Protocol):
    name: str
    def load(self, split, limit, seed) -> tuple[list[Example], dict]: ...   # dict = task_context (e.g. label set)
    def build_item_block(self, example, task_context) -> str: ...           # task text WITHOUT affect cue
    def parse(self, raw_output, example, task_context) -> Parsed: ...
    def score(self, parsed, example) -> dict: ...   # returns per-item metric signals (see §4)
    def max_new_tokens(self) -> int: ...            # tasks differ: 32 for labels, 512+ for IFEval
```

Shared prompt assembler (moves out of `banking77.build_prompt`):
```
[TASK]\n{task.header}\n\n{build_affect_block(condition)}\n\n{task.build_item_block(...)}\n\n[ANSWER]\n
```

### 3.2 Task adapters

- `tasks/banking77.py` — refactor the existing module to the `Task` interface
  (behavior-preserving; the existing tests must still pass).
- `tasks/ifeval.py` — loads IFEval, vendors the constraint checkers from the
  official `instruction_following_eval` verifier, scores strict + loose
  prompt-level and per-instruction satisfaction.
- `tasks/ruler_niah.py` — generates synthetic haystacks with a controlled needle
  at parameterized depths; scores exact-span retrieval and records needle depth
  + context length per item for the position-decay analysis.

### 3.3 Generic runner `run_stage2.py`

Adapt `run_stage1.py`:
- `--task {banking77,ifeval,ruler_niah}` selects the adapter via a registry.
- `--model`, `--engine`, condition/baseline handling, chat rendering, output
  writing all reuse Stage 1 unchanged.
- `max_new_tokens` comes from the task, not a global default.
- Keep Stage 1 fully runnable (either keep `run_stage1.py` as a thin shim that
  calls `run_stage2 --task banking77`, or leave it untouched for reproducibility).

`generation.py` needs no change. `affect.py` gains paraphrase variants (§5).

## 4. Expanded metrics (the point of Stage 2)

Extend `metrics.py` additively. The existing `paired_accuracy_deltas` already
computes paired bootstrap deltas over a binary per-item signal — generalize it to
`paired_deltas(rows, signal_key, baselines, ...)` so every new binary metric gets
the same paired-CI + wins/ties/losses treatment vs. both neutral baselines.

Per-item signals each task's `score()` emits (task-appropriate subset):

| Metric | banking77 | IFEval | RULER NIAH |
|---|---|---|---|
| correct / task score | label exact-match | strict & loose constraint satisfaction | span retrieved |
| omission rate | unparsed / refused | constraint silently dropped | "not found" / missed needle |
| format-valid-but-wrong | valid label, wrong | well-formed, violates a constraint | plausible value, wrong |
| hallucination | invented (out-of-set) label | invented content vs. required source | wrong/invented value |
| refusal rate | explicit refusal string | explicit refusal | explicit refusal |
| verbosity / cost | output tokens | output tokens | output tokens |
| position sensitivity | — | — | accuracy × needle depth (key output) |

Reported per condition and as paired deltas vs. `neutral_empty` **and**
`neutral_length_matched`. Refusal-string and "invented label" detection are new
small helpers.

## 5. Prompt-robustness fix (required by H3)

Stage 1 used **one phrasing per condition**, so a null could be "this wording"
rather than "this construct." Add to `affect.py`:
- 2–3 paraphrases per affect condition (`AffectCondition.variants: tuple[str, ...]`).
- **Dev/test split discipline:** run all paraphrases on a locked **dev** slice,
  select the best-performing paraphrase per condition (or report the full
  paraphrase spread), then run the *locked* prompts on the held-out **test**
  slice. No best-of-many test claims. This is what the research plan's statistical
  section already requires and Stage 1 omitted.

## 6. Models

- Primary: `Qwen/Qwen3-8B` (matches Stage 1 for continuity).
- Second model to test model-specificity: `Qwen/Qwen3-14B` (same family, isolates
  scale) or `meta-llama/Llama-3.1-8B-Instruct` (isolates family). Recommend
  Qwen3-14B first — cleanest scale comparison.

Hardware is not a constraint: the dev box has 8×A100-40GB. 14B runs on one GPU;
long-context RULER at 16k benefits from vLLM's paged attention. Run models
sequentially to keep memory simple.

## 7. Statistics

Reuse Stage 1's paired cluster bootstrap. Additions:
- Paired deltas for every new metric (not just accuracy), vs. both baselines.
- **Multiple-comparison control:** Stage 2 has conditions × tasks × metrics ×
  models comparisons. Apply Holm–Bonferroni (or Benjamini–Hochberg FDR) across the
  primary family of tests and report adjusted significance. State the primary
  hypothesis family up front to avoid post-hoc inflation.
- Report model × condition and task × condition interactions.

## 8. Sequencing

1. **Refactor to `Task` interface**; port banking77; keep existing tests green.
   (Structural, no new science — do first so the two new tasks are small.)
2. **IFEval adapter + expanded metrics module.** First real mechanism-matched
   test. Cheap, fast to iterate the metric definitions on.
3. **RULER NIAH adapter** with depth sweep. The core-hypothesis probe.
4. **Paraphrase variants + dev/test lock.** Apply across all three tasks.
5. **Second model** (Qwen3-14B) across all tasks.
6. **Analysis + write-up** into `docs/article-outline.md` §6.

## 9. Decision gate

- If IFEval and RULER **still** show nothing past length-matched neutral across
  two models and the new metrics → strong, publishable **negative result**
  (research-plan Outcome D) with a reusable benchmark + control framework, and
  Stage 3 reframes cleanly as "does the affect direction move behavior
  *internally* even when prompts do not?"
- If something moves — especially omission or position-decay on RULER, or
  constraint satisfaction on IFEval — that is the signal to (a) run the deferred
  breadth tasks and (b) proceed to the Stage 3 activation study.

Either outcome is a clean result. What we avoid is scaling breadth or building
steering infrastructure before the prompt hypothesis has had a fair test on tasks
that could actually reveal the effect.

## 10. Outputs & repro

Per (task, model) run, mirror Stage 1's artifact set under
`runs/stage2_<task>_<model>/`: `prompts.jsonl`, `predictions.jsonl`,
`summary.csv` (extended columns), `paired_deltas.csv` (per metric),
`confusions.jsonl` (task-appropriate), `metadata.json`, `run_stats.json`,
`requirements.freeze.txt`, `nvidia-smi.txt`. All runs live on EC2 at
`/home/ec2-user/code/personal/emotion-llm-capabilities/runs/`.

---

# 11. Post-execution addendum (2026-08-06): results + next step

## 11.1 What Stage 2 actually found

Executed exactly as designed: `ifeval` and `ruler_niah`, each on `Qwen/Qwen3-8B`
and `Qwen/Qwen3-14B`, dev-variant paraphrase selection, locked test splits, full
reliability panel, and Holm–Bonferroni global correction over a 48-test primary
family. Aggregate:
`runs/stage2_locked_test_aggregate/{review_summary.json,combined_summary.csv,combined_paired_deltas.csv}`.

**Headline: `globally_significant_primary_tests: []`.** No affect condition beats
either neutral baseline on any primary metric after correction, on either model.

Three findings underneath that:

1. **IFEval is a clean null.** strict-prompt ≈ 0.83 (8B) / 0.85 (14B), flat across
   all 8 conditions. Largest paired delta is a fraction of a point; all
   Holm-global p = 1.0.

2. **RULER did not test the hypothesis — it was a ceiling task.** Stratifying the
   locked runs by (context_length × depth × mode):
   - **8B: 100% correct in all 18 cells.** Zero variance to move.
   - **14B: at ceiling except non-monotonic noise** (e.g. 8k/single/depth-0.1 =
     0.31 while 16k/single/depth-0.1 = 1.0 — variance, not position decay).
   The best raw blip, `calm_meticulous` +0.11 vs neutral_empty (raw p=0.008),
   collapses to Holm-global p=0.375. **With no decay in the baseline, an affect
   effect on vigilance is structurally unobservable.** This is the reason to
   re-run, and it is a task-construction problem, not a hypothesis result.

   *Why RULER is too easy (root cause, from `tasks/ruler_niah.py`):* the needle is
   a screaming-caps `IMPORTANT EXCEPTION RECORD` line, filler is perfectly uniform
   ("Routine record …; status normal; no exception noted"), the answer is a unique
   regex-matchable `NIAH-XXXX-XXXX` code, context caps at 16k (well inside Qwen3's
   native window), and multi mode uses only 3 needles. Every one of those makes
   retrieval trivial.

3. **The one robust, replicated effect is verbosity — a cost, not a benefit.**
   Affect cues (esp. `positive_pride`, `calm_meticulous`) reliably raise output
   tokens: **+66** (pride, 8B IFEval) and **+40–47** (calm, 14B IFEval), p≈0,
   both models. So affective framing *does* causally change behavior — it makes
   the model talk more without getting more right. Caveat: `output_tokens` is a
   secondary metric, outside the corrected primary family, so report it as a
   descriptive effect (the effect sizes are large enough to stand without
   correction). This is Outcome B/D territory in `docs/article-outline.md`.

**Verdict:** primary-metric null holds across two tasks and two models, but RULER's
null is uninformative. Before declaring the prompt hypothesis settled, RULER must
be re-run at a difficulty where the target failure mode (position-wise vigilance
decay) actually appears.

## 11.2 Step A — Harden RULER (the executable next step)

Goal: move baseline (neutral) RULER accuracy off the 1.0 ceiling into a **0.5–0.8
band** so there is headroom for an affect effect to be visible, *then* re-run the
locked dev→test affect comparison on the harder task.

The difficulty knobs are already CLI-exposed on `run_stage2.py`
(`--ruler-context-lengths`, `--ruler-depths`, `--ruler-modes`,
`--ruler-samples-per-cell`) — most of Step A is a calibration sweep plus a few
targeted adapter changes, not a rewrite.

**A1. Push context length past the trivial range.** Add 32k / 65k / 131k to
`--ruler-context-lengths`. Blocker to fix first (in code): `VLLMGenerator`
(`emotion_stage1/generation.py`) constructs `LLM(model, dtype, trust_remote_code)`
with **no `max_model_len`**, so it inherits the model default and cannot serve
>32k. Add a `max_model_len` (and likely `gpu_memory_utilization` /
`enforce_eager`) knob to `GenerationConfig` and thread a `--max-model-len` flag
through `run_stage2.py`. Qwen3's native window is 32k; going to 128k requires YaRN
rope scaling (`rope_scaling` in the vLLM call) — decide explicitly whether to (i)
cap at 32k and get difficulty from harder needles instead, or (ii) enable YaRN and
report that long-context accuracy is partly a rope-scaling artifact. **Recommend
(i) first** — cap at 32k, make the *task* harder — because it isolates vigilance
from rope-scaling confounds.

**Execution decision (2026-08-06): use option (ii).** Enable YaRN with a 131,072
token vLLM window, Qwen3's configured original window of 40,960 tokens, and a
derived factor of 3.2. Calibrate at 32,768 / 65,536 / 131,072 tokens and report
the rope-scaling confound explicitly. Use five depths (0.1 / 0.3 / 0.5 / 0.7 /
0.9), all three modes (single / multi / aggregate), 12 distractors, 6 keys in
multi and aggregate modes, and a 1,024-token prompt reserve as the first
neutral-only calibration configuration. Three held-out samples per cell yield
135 items per arm.

**Calibration log.** V1 used 12 distractors, 6 multi keys, and 6 aggregate
matches. It was rejected: neutral exact accuracy was 0.267, multi output was
truncated at the old 64-token cap in every cell, aggregate accuracy was 0, and
depth also changed the synthetic document seed instead of only target position.
V2 corrects the cap to 128, holds target/filler identity constant across depth,
and sweeps an easier hard-task setting of 6 distractors, 5 multi keys, and 3
aggregate matches. V1 artifacts remain preserved.

V2 was also rejected at 0.411 overall. V3 made needle identity invariant to
difficulty knobs and reduced distractors to 3; it reached 0.467. V4 changed
only aggregate matches from 3 to 2 and reused V3's unchanged single/multi
cells. The combined neutral baseline is 0.511, with context accuracy
0.867 / 0.400 / 0.267 at 32k / 64k / 128k and no truncation. Lock the V4
configuration: 3 distractors, 5 multi keys, and 2 aggregate matches.

The calibrated position curve is 0.611 / 0.500 / 0.444 / 0.389 / 0.611. It
shows clear degradation through depth 0.7 followed by an end-of-context recency
rebound at 0.9. Report that U-shaped position effect directly; do not describe
it as strictly monotonic depth decay.

**A2. Make retrieval hard at fixed context (the higher-value lever).** Edit
`tasks/ruler_niah.py`:
- **Distractor needles:** insert several *decoy* `IMPORTANT EXCEPTION RECORD`
  lines with the same surface form but wrong case-ids / codes, so the model must
  disambiguate rather than pattern-match the one salient line. This alone should
  break the ceiling.
- **De-salience the needle:** drop the all-caps `IMPORTANT EXCEPTION` framing;
  phrase the needle like the surrounding routine records so it does not pop out.
- **Heterogeneous filler:** vary the routine-record wording so the needle is not
  the only irregular line.
- **Scale multi mode:** raise the multi-key count (e.g. 5–8 needles) and add an
  aggregation variant ("how many exception records mention X"), which taxes
  completeness/omission — the metric affect is theorized to move.
- Keep depth ∈ {0.1, 0.5, 0.9} (already the right vigilance-vs-position axis) and
  consider adding 0.3 / 0.7 once there is headroom to resolve a depth curve.

**A3. Calibrate before committing.** Run neutral-only (`--split dev`, a subset of
conditions or just the two baselines) across the candidate difficulty grid, read
accuracy-by-cell, and pick the config where neutral lands in 0.5–0.8 with a
*visible depth gradient* (deeper needle → lower accuracy). Lock that config. Log
what was swept and chosen — no silent difficulty tuning.

**A4. Re-run the full affect comparison on hardened RULER.** Same protocol as the
locked Stage 2 RULER runs (dev-variant selection → locked test, both models, full
reliability panel, Holm-global correction). Raise `--ruler-samples-per-cell` so
per-cell n is large enough to detect a plausible effect at the new baseline (do a
quick power sketch: to detect ~5–10 pt at p≈0.5 baseline you want on the order of
100+ items per arm, up from 72). New artifacts under
`runs/stage2_ruler_niah_hard_<model>_{dev_variants,test_locked}/` and a refreshed
`stage2_locked_test_aggregate`.

**A4 execution result (2026-08-07).** The locked held-out runs completed for
both models with 135 items per arm, 1,080 predictions per model, and zero
truncated outputs:

- `Qwen/Qwen3-8B`: neutral accuracy was 0.556 (`neutral_empty`) / 0.607
  (`neutral_length_matched`). Empty-neutral context accuracy was 0.689 / 0.644 /
  0.333 at 32k / 64k / 128k. Its depth curve was 0.630 / 0.556 / 0.444 / 0.481 /
  0.667, preserving the degradation plus end-recency rebound seen in
  calibration.
- `Qwen/Qwen3-14B`: neutral accuracy was 0.948 / 0.956. Accuracy remained
  0.933 or higher at every context length, so this model is still near ceiling
  and is not a sensitive test of the prompt effect under the shared locked
  configuration.

The four-run aggregate (the two existing locked IFEval runs plus these two hard
RULER runs) is
`runs/stage2_locked_test_aggregate_hard_ruler_128k/`. None of the 48 primary
tests survive global Holm correction. The smallest global p-value is 0.375 for
an adverse 8B effect: `high_stakes_audit` is -0.059 versus the length-matched
neutral control. The apparent 8B `negative_failure_avoidance` gain is +0.074
versus empty neutral (raw p=0.021, global p=0.9997) but only +0.022 versus the
required length-matched control (raw p=0.549).

**Locked verdict:** no affect condition improves the primary metric after the
required control comparison and multiplicity correction. This is a diagnostic
negative result for Qwen3-8B. Qwen3-14B also shows no effect, but its ceiling
means the run should not be used as strong evidence that the effect is absent at
that scale. A future two-model claim would require a separately pre-registered
14B difficulty calibration rather than post-hoc tuning on these test results.

**Secondary reliability pattern (8B, hardened RULER, uncorrected).** Beyond the
primary null, the 8B affect arms trend *adversely* on the reliability panel
against the length-matched control — a coherent tradeoff signal, not just noise
around zero. Largest nominal effects (vs `neutral_length_matched`, raw p, outside
the corrected primary family so descriptive only): `calm_meticulous`
retrieval_rate −0.087 (p=0.0005) and hallucination +0.081 (p=0.003);
`high_stakes_audit` hallucination +0.067 (p=0.004) and correct −0.059 (p=0.008);
`boredom_antidote_repetitive` correct −0.059 (p=0.022). Direction is consistent:
under affect framing the 8B model retrieves the true value slightly less and
emits wrong/invented codes slightly more. Combined with the replicated
verbosity increase (IFEval), the honest one-line story is **"affective framing
does not help tedious-task reliability and, at 8B, mildly hurts long-context
retrieval while raising cost"** — Outcome C-leaning within the article's
tradeoff framing. Report these as descriptive/secondary; they are not
Holm-global significant.

**Decision at end of Step A:**
- **Still null on hardened RULER (both models)** → the prompt hypothesis is now
  *fairly* tested and settled: prompt-level affect framing does not improve
  tedious-task reliability; its only reproducible effect is increased verbosity.
  Strong negative-result paper + reusable framework. Proceed to Step B.
- **Signal appears** (affect moves accuracy / omission / position-decay past
  length-matched neutral, post-correction) → this is the first positive result;
  (a) run the deferred breadth tasks to check generality and (b) still proceed to
  Step B, now to test whether the *internal* direction mediates the prompt effect.

Either way Step B (Stage 3) follows; the RULER result only changes its framing.

## 11.3 Step B — Bridge to Stage 3 (activation steering), reframed

The Stage 2 result sharpens Stage 3's question. Old framing: "do affect vectors
help?" New framing, given the prompt-level null:

> **Prompt-level affect framing is a weak lever — it only moves verbosity. Does a
> direct internal affect/diligence *direction* move task behavior when prompts do
> not?** And where the prompt did move something (verbosity), does steering the
> corresponding direction reproduce or dissociate that?

This is the mechanistically interesting question and the strongest version of the
article. Concretely, Stage 3 setup (detailed plan to be its own `docs/stage3-plan.md`):

- **Engine reality:** activation capture + steering needs residual-stream hooks —
  the **`transformers`** backend, not vLLM. `generation.py` already has a
  `TransformersGenerator`; Stage 3 extends it with forward hooks for read
  (hidden-state collection) and write (add a steering vector at chosen layers).
  vLLM stays the path for the prompt-only RULER/IFEval re-runs.
- **Vectors:** reuse the existing concept vocabulary and story-prompt builder
  (`emotion_stage1/affect.py`: `EMOTION_VECTOR_CONCEPTS`,
  `write_emotion_story_prompts.py`) to generate matched contrastive vignettes,
  collect activations, and take difference-in-means directions per concept pair
  (calm/desperate, focused/distracted, patient/rushed, responsible/careless,
  curious/bored) — the research-plan §Stage 3 procedure.
- **Causal tests (reuse the Stage 2 harness as the eval):** the hardened RULER +
  IFEval tasks become the downstream eval for steered generations. Run the
  plan's crossed design — neutral prompt + affect vector, affect prompt + opposite
  vector, random-vector control, shuffled-label control, dose-response sweep,
  layer sweep — scoring with the *same* reliability panel and paired-bootstrap +
  Holm-global machinery already built. This reuse is the payoff of the Stage 2
  investment.
- **Gate into Step B:** start with probe validation (do the directions linearly
  separate held-out vignettes without keyword leakage?) before any steering eval,
  so a null steering result is interpretable (direction exists but doesn't move
  behavior) vs. uninterpretable (no direction found).

## 11.4 Sequencing (supersedes §8 for remaining work)

1. **A1** — add `max_model_len` (+ mem/eager) knob to `GenerationConfig` /
   `run_stage2.py`; verify a >16k neutral RULER run executes.
2. **A2** — harden `tasks/ruler_niah.py` (distractors, de-salience, heterogeneous
   filler, larger multi-key). Update `tests/test_stage2.py` for the new needle
   structure.
3. **A3** — neutral-only difficulty calibration sweep; lock the config in the 0.5–
   0.8 band with a visible depth gradient.
4. **A4** — full dev→test affect comparison on hardened RULER, both models;
   refresh aggregate; record verdict.
5. **B** — write `docs/stage3-plan.md`; extend `TransformersGenerator` with
   hooks; probe validation → steering eval on the Stage 2 tasks.

## 11.5 What we are still NOT doing

- **No breadth sweep yet** (CUAD / table-QA / spreadsheet / code-repair). Gated on
  a positive signal from hardened RULER, per §9. Two tasks × two models already
  give a well-powered primary-metric null; more accuracy-only tasks would add cost
  without changing the conclusion.
- **No fine-tuning / adapter pilot** (research-plan Stage 4). Only if Steps A/B
  produce a repeatable effect worth distilling.
- **No difficulty tuning against the affect conditions.** RULER hardening is
  calibrated on *neutral only* (A3) and then locked, so the affect comparison in
  A4 is not tuned to produce an effect.
