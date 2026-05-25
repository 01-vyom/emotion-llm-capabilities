# Emotion LLM Capabilities

Research code and notes for testing whether affective framing and emotion-like
internal control signals can improve LLM reliability on tedious, low-salience
tasks.

The current implementation is **Stage 1**: a prompt-only pilot using an
open-weight 8B-scale model and an objectively scored repetitive classification
benchmark.

## Stage 1

- Model: `Qwen/Qwen3-8B`
- Benchmark: `PolyAI/banking77`
- Task: classify customer-service messages into one of 77 intent labels
- Conditions: neutral, length-matched neutral, positive pride, failure
  avoidance, audit stakes, extreme stakes, calm meticulous, and
  boredom-antidote
- Metrics: accuracy, parse rate, macro-F1, bootstrap confidence intervals, and
  confusion counts

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

- [Research plan](docs/research-plan.md)
- [Literature survey](docs/literature-survey.md)
- [Article outline](docs/article-outline.md)
- [Stage 1 implementation notes](emotion_stage1/README.md)

## Core Hypothesis

Moderate affective framing may improve reliability on repetitive, exacting tasks
by increasing task salience and verification-like behavior. Extreme stress may
instead increase shortcutting, overconfidence, refusal drift, sycophancy, or
unsafe compliance. The project treats emotion as a functional control signal,
not as evidence that models have subjective feelings.

## Later Stages

Stage 2 will broaden the prompt-only benchmark suite. Stage 3 will test
activation-level emotion or diligence vectors using contrastive activation
addition / representation engineering methods inspired by the Anthropic emotion
concepts paper.
