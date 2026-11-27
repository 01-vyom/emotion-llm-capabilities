# Stage 1: Affective Prompting Pilot

This is the prompt-only Stage 1 experiment for testing whether affective cues
change performance on a tedious, objective task.

## Choices

- Model: `Qwen/Qwen3-8B`
  - 8B-scale, Apache-2.0, current open-weight family, appropriate for one A100.
- Dataset: `PolyAI/banking77`
  - 13,083 banking support queries, 77 fine-grained intents, CC BY 4.0.
  - This is a repetitive, low-novelty classification task with objective scoring.

The linked Anthropic emotion paper found emotion concepts by generating stories
where characters experienced target emotions, then extracting activation
directions from model residual streams. This Stage 1 code does not collect
activations yet, but `affect.py` includes the matching emotion concept set and
`write_emotion_story_prompts.py` writes synthetic story prompts for the later
activation-vector stage.

## Setup

```bash
python3 -m venv .venv-emotion
source .venv-emotion/bin/activate
pip install -r emotion_stage1/requirements.txt
```

## Dry Run

Writes prompts and metadata without loading the model:

```bash
python -m emotion_stage1.run_stage1 \
  --dry-run-prompts \
  --limit 20 \
  --output-dir runs/stage1_banking77_dry
```

If the Hugging Face `datasets` package is not installed yet, use the built-in
fixture just to verify the harness:

```bash
python -m emotion_stage1.run_stage1 \
  --dry-run-prompts \
  --fixture-smoke-test \
  --output-dir runs/stage1_fixture_dry
```

## A100 Run

```bash
python -m emotion_stage1.run_stage1 \
  --engine vllm \
  --model Qwen/Qwen3-8B \
  --limit 500 \
  --batch-size 32 \
  --output-dir runs/stage1_banking77_qwen3_8b
```

The runner defaults to `--prompt-format chat --disable-thinking`, which is the
right setting for Qwen3-style chat models when the output needs to be a parseable
intent label.

Outputs:

- `prompts.jsonl`: exact prompts sent to the model.
- `predictions.jsonl`: raw outputs, parsed labels, correctness.
- `summary.csv`: accuracy, parse rate, macro-F1, bootstrap CI per condition.
- `paired_deltas.csv`: paired accuracy deltas against both neutral controls,
  with bootstrap CIs and win/tie/loss counts.
- `confusions.jsonl`: most common wrong gold/prediction pairs.
- `metadata.json`: model, dataset, split, seed, and condition list.

## Default Affect Conditions

- `neutral_empty`
- `neutral_length_matched`
- `positive_pride`
- `negative_failure_avoidance`
- `high_stakes_audit`
- `high_stakes_extreme`
- `calm_meticulous`
- `boredom_antidote_repetitive`

The key control is `neutral_length_matched`. If it matches or beats the affective
conditions, the honest conclusion is that emotion was not the active ingredient.

## Future Vector Stimuli

To write Anthropic-style story prompts for Stage 3 activation work:

```bash
python -m emotion_stage1.write_emotion_story_prompts \
  --replicates 3 \
  --output runs/emotion_story_prompts.jsonl
```
