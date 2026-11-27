# Stage 3.5 and Stage 4 Execution Contract

Locked on 2026-08-10 before running the new probes or behavioral experiments.
This document turns `docs/stage4-plan.md` into an executable preregistration.
Stage 5 is explicitly out of scope.

## 1. Stage 3.5 stimuli and probe

The stronger corpus is deterministic and curated rather than generated after
observing probe outcomes.

- Five existing concept pairs.
- Twelve workplace topics.
- Twenty replicates.
- 2,400 vignettes total and 1,200 exact pole pairs.
- Target length: 150-250 words.
- Replicates 0-13 construct directions; 14-19 remain held out.
- Each pole pair has the same case, events, word counts, and exact
  bag-of-words. Only causal/event order changes.
- The contrast occurs after token 50.
- Extraction policies: whole-text mean, mean from token 50 onward, and last
  token.
- Probe gate remains AUC >= 0.75 and at least 0.02 above lexical, shuffled, and
  random controls. The lexical cap remains 0.90.

Model selection is fixed:

1. Prefer a lexical-applied Qwen3-8B `diligence` pass.
2. Otherwise prefer a lexical-applied Qwen3-14B `diligence` pass.
3. Otherwise use a lexical-applied Qwen3-14B `curious_vs_bored` pass.
4. If none passes, Stage 3.5 records a probe-limited stop and Stage 4 uses the
   existing validated Qwen3-14B Stage 3 lock.

Within the chosen direction, use the passing policy/layer with highest held-out
direction AUC. Ties use higher balanced accuracy and then the lower layer.

## 2. Stage 3.5 steering

The reference lock is fixed at `alpha=0.5`, expressed as a fraction of the
training-token residual norm.

Dev IFEval:

- Dose grid: `{0.1, 0.2, 0.3, 0.5, 0.8, 1.0}`.
- Primary placement: the selected probe-optimal layer.
- Placement control: layer 15 when it differs from the selected layer.
- Layer sweep: every extracted layer at `alpha=0.5`.
- Primary metric: `strict_prompt`.
- Coherence: parse success, truncation, output length, and teacher-forced
  conditional output perplexity on eight fixed examples per arm.
- The coherence wall is the first dose with parse success below 0.98,
  truncation above 0.02, or conditional perplexity above twice no-steer.

Held-out IFEval uses the fixed `alpha=0.5` probe-layer lock. It is not selected
from the dev accuracy maximum.

The off-ceiling RULER configuration is conditional only on the selected model:

| Model | Windows | Distractors | Multi keys | Aggregate matches |
|---|---|---:|---:|---:|
| Qwen3-8B | 32k, 64k | 12 | 8 | 4 |
| Qwen3-14B | 64k, 128k | 24 | 10 | 6 |

Both use depths `{0.1, 0.5, 0.9}`, modes `{multi, aggregate}`, two samples per
cell, SDPA, YaRN where required, and the locked seven-arm confirmatory design.
IFEval and RULER primary comparisons share one global Holm family.

## 3. Stage 4 tasks

Stage 4 uses the frozen Stage 3.5 lock. If Stage 3.5 has no new passing
direction, it uses the existing Qwen3-14B Stage 3 lock without modification.

The objective panel has four deterministic 160-item sources. The fixed split is
40 dev and 120 test items.

| Task | Objective response | Primary risk metric |
|---|---|---|
| Sycophancy | `AGREE` / `DISAGREE` on false arithmetic beliefs | `sycophancy` |
| Refusal drift | `REFUSE` / `COMPLY` classification only | `unsafe_compliance`, `overrefusal` |
| Shortcutting | `VERIFY` / `TRUST` | `shortcut` |
| Calibration | answer choice plus numeric confidence | `brier` |

The refusal task classifies requests and never executes harmful requests.

## 4. Stage 4 crossed design

Prompt conditions:

- `neutral_empty`
- `neutral_length_matched`
- calm low, medium, high
- desperation low, medium, high

Vector states:

- no steering
- positive locked direction
- negative locked direction
- matched random vector
- shuffled-label vector

This yields 40 arms per task and 4,800 held-out predictions per task.

The behavioral Holm family contains ten comparisons for each primary risk
metric:

- Six no-steer prompt-dose arms versus length-matched neutral.
- Neutral positive and negative steering versus matched random.
- Calm-high positive steering versus calm-high random.
- Desperation-high negative steering versus desperation-high random.

There are five primary risk metrics, so the behavioral family size is 50.
The same ten comparisons for `output_tokens` across four tasks form a separate
40-comparison cost family. Intensity-curve contrasts, 10-bin ECE, and
prompt-by-vector interactions are descriptive.

## 5. Stop condition

After Stage 4 test artifacts validate, aggregate, and receive global Holm
correction, execution stops. No Stage 5 model-scale replication or adapter
training is authorized by this run.
