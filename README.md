# Emotion isn't the active ingredient: cheap controls explain away the emotional effects in Qwen Models

[Report](https://docs.google.com/document/d/1NqnWK7fCpIGDUoMf4hfkzcF31tjFYP1UjbFVRLBIjbE/edit?usp=sharing)

I tested how emotional prompts and activation steering affect instruction
following, refusal decisions, and agreement with false claims in Qwen3 models
with 8, 14, and 32 billion parameters.

Did asking Qwen3-32B to stay calm help it recognize false claims? On the same
600 claims, it agreed with 21% under the calm cue and 30.5% under a
length-matched neutral cue. In an exploratory comparison, it agreed with just
19% when the cue was omitted. The apparent benefit depended on the baseline.

## Original hypothesis

I expected a moderate emotional cue to help a model pay attention and check its
work on repetitive tasks. Stronger pressure might instead encourage shortcuts,
overconfidence, agreement with false claims, or mistakes about which requests
to refuse. The experiments measured changes in model behavior.

## Research stages

### Stage 1 · Classifying banking requests

Qwen3-8B classified 500 customer-service messages into 77 Banking77 intent
labels under eight prompt conditions. Accuracy ranged from 62.0% to 63.6%.
Many errors involved easily confused labels, which motivated a more direct
test of whether the model followed instructions. [Implementation notes](emotion_stage1/README.md)

### Stage 2 · Following instructions and retrieving details

I tested Qwen3-8B and 14B on IFEval, which checks whether an answer follows
every instruction, and RULER, which asks the model to retrieve details from
long contexts. Some cues produced longer answers. None improved a primary
score after correction for multiple comparisons.

The suite includes single-key, multi-key, and aggregate retrieval, with a
harder 32k/64k/128k YaRN configuration. It tracks omissions, answers in the
right format with the wrong content, hallucinations, refusals, parsing, and
output length. I selected prompt variants on development data, locked them
before testing, and used paired bootstrap intervals, exact paired sign tests,
and Holm correction. [Plan](docs/stage2-plan.md) · [Runbook](emotion_stage2/README.md)

### Stage 3 · Steering internal activations

A probe could distinguish curious from bored stories in held-out data at 14B.
Steering along that direction did not improve held-out IFEval or harder
32k/64k retrieval. In a follow-up (Stage 3.5), I rebuilt the stories as
word-matched pairs. On 433 IFEval prompts at 8B, diligent steering scored
0.9 percentage points above random steering (95% CI −2.5 to +4.6).
[Execution record](docs/stage3-plan.md) · [Runbook](emotion_stage3/README.md)

### Stage 4 · Testing safety judgments

I shifted from accuracy to agreement with false claims, refusal decisions,
shortcuts, and confidence calibration. At 8B, unsafe compliance and
shortcutting both stayed at 0% in every condition. That made it impossible to
tell whether a cue could reduce either behavior. [Stage 3.5 and 4 methods](docs/stage35-stage4-execution.md)

### Stage 5 · Making the safety tasks harder

I revised the 8B tasks so there was room for error before adding a cue or
steering. The model now sometimes agreed with false claims, approved harmful
requests, and trusted an unchecked answer. It still refused 0% of harmless
requests at baseline, so this run could not test a reduction in unnecessary
refusals. [Methods](docs/stage5-execution.md)

### Stage 6 · Checking the controls across model sizes

I reanalyzed the 8B steering results against no steering, a random
direction, and a direction built from shuffled story labels. The
shuffled direction reproduced the apparent increase in shortcutting. I then
tested safety judgments across Qwen3-8B, 14B, and 32B, with a prompt-only
comparison on Claude Sonnet 5. [Methods](docs/stage6-execution.md)

### Stage 7 · Calibrating difficulty for each model

I chose a different mix of items for each Qwen3 model, aiming for a baseline
error rate near 30%. At 8B, the calm cue increased agreement with easy false
claims and reduced agreement with harder ones, both relative to a neutral cue.
On the refusal task, urgency increased REFUSE judgments by 11.9 percentage
points for harmful requests and 27.7 points for harmless ones. The different
item mixes limit comparisons across model sizes. [Methods](docs/stage7-execution.md)

## Run the Banking77 pilot

- Model: `Qwen/Qwen3-8B`
- Benchmark: `PolyAI/banking77`
- Task: classify customer-service messages into one of 77 intent labels
- Conditions: neutral, length-matched neutral, positive pride, failure
  avoidance, audit stakes, extreme stakes, calm meticulous, and
  boredom-antidote
- Metrics: accuracy, parse rate, macro-F1, paired bootstrap confidence
  intervals, win/tie/loss counts, and confusion counts

Run:

```bash
python3 -m venv .venv-emotion
source .venv-emotion/bin/activate
pip install -r emotion_stage1/requirements.txt

python -m emotion_stage1.run_stage1 \
  --engine vllm \
  --model Qwen/Qwen3-8B \
  --limit 500 \
  --batch-size 32 \
  --output-dir runs/stage1_banking77_qwen3_8b
```

Smoke test without ML dependencies:

```bash
python -m emotion_stage1.run_stage1 \
  --dry-run-prompts \
  --fixture-smoke-test \
  --output-dir runs/stage1_fixture_dry
```

## Research docs

- [Research plan](docs/research-plan.md)
- [Literature survey](docs/literature-survey.md)
- [Article outline](docs/article-outline.md)

### Rebuilding the write-up and checks

The write-up contains the methods, examples and results needed to read it on its
own. Repository commands and artifact locations are kept here:

- `python docs/build_mats_writeup.py` renders the Markdown to self-contained HTML
  and, when Chrome is available, PDF.
- `python docs/figures/make_mats_figures.py` rebuilds the overview and numbered
  figures from `docs/figures/data/`. The inputs include run aggregates and
  follow-up analyses of existing outputs: truncation and repetition-loop checks,
  the random-versus-no-steering bootstrap interval, text baselines, cue
  comparisons, and example draws.
- `python docs/figures/bigram_baseline.py <vignettes.jsonl>` reruns the text-only
  baseline on the steering stories.
- `docs/figures/data/changed_examples_seed2026.json` contains the sampled changed
  items and flip counts; `docs/figures/data/stage35_ifeval_truncated_loops.csv`
  contains the repetition-loop check.
- Plans and preregistered test sets are in `docs/stage*-plan.md` and
  `docs/stage7-execution.md`.
