# Emotion LLM Capabilities

Research code and notes for testing whether affective framing and emotion-like
internal control signals can improve LLM reliability on tedious, low-salience
tasks.

The repository contains the completed **Stage 1** classification pilot,
**Stage 2** mechanism-matched IFEval and RULER NIAH suite, and **Stage 3**
activation-direction and steering study.

## Stage 1

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

## Research Docs

- **[MATS write-up](docs/mats-writeup.md)** (canonical report; HTML/PDF via `python docs/build_mats_writeup.py`, figures via `python docs/figures/make_mats_figures.py`)
- [Research plan](docs/research-plan.md)
- [Literature survey](docs/literature-survey.md)
- [Article outline](docs/article-outline.md)
- [Stage 1 implementation notes](emotion_stage1/README.md)
- [Stage 2 implementation notes](emotion_stage2/README.md)
- [Stage 2 plan](docs/stage2-plan.md)
- [Stage 3 implementation notes](emotion_stage3/README.md)
- [Stage 3 plan and execution record](docs/stage3-plan.md)

## Core Hypothesis

Moderate affective framing may improve reliability on repetitive, exacting tasks
by increasing task salience and verification-like behavior. Extreme stress may
instead increase shortcutting, overconfidence, refusal drift, sycophancy, or
unsafe compliance. The project treats emotion as a functional control signal,
not as evidence that models have subjective feelings.

## Stage 2

Stage 2 adds:

- official-verifier IFEval constraint compliance
- deterministic RULER-style single-key, multi-key, and aggregate retrieval,
  including a hardened 32k/64k/128k YaRN configuration
- omission, format-valid-but-wrong, hallucination, refusal, parse, and output
  token signals
- paired bootstrap deltas, exact paired sign tests, and Holm correction
- dev-only prompt variant selection with locked held-out test prompts

See [the Stage 2 runbook](emotion_stage2/README.md) for commands and artifact
validation.

## Stage 3

Stage 3 extracted leakage-controlled affect directions and tested locked
activation addition. One Qwen3-14B `curious_vs_bored` direction passed the
corrected held-out probe gate, but its dev-selected steering configuration did
not improve held-out IFEval or hardened 32k/64k RULER. No confirmatory
comparison survived global Holm correction.

See [the Stage 3 execution record](docs/stage3-plan.md) and
[Stage 3 runbook](emotion_stage3/README.md).
