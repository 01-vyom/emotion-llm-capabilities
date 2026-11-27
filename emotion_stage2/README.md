# Stage 2: Mechanism-Matched Suite

Stage 2 tests affective prompt framing on tasks that exercise omission,
constraint fidelity, and position-sensitive rare-detail retrieval.

## Tasks

- `ifeval`: `google/IFEval`, scored by the vendored official Google Research
  strict and loose verifiers. Its 4,096-token generation ceiling accommodates
  the benchmark's long-form constraints; inspect `output_truncated` before
  accepting a run.
- `ruler_niah`: deterministic synthetic single-key, multi-key, and aggregate
  retrieval. The hardened configuration adds de-salient targets, heterogeneous
  filler, and matched distractor records.
- `banking77`: compatibility adapter for the completed Stage 1 task.

## Prompt Lock Protocol

Run all variants on the deterministic dev split:

```bash
python -m emotion_stage2.run_stage2 \
  --task ifeval \
  --model Qwen/Qwen3-8B \
  --split dev \
  --all-variants \
  --output-dir runs/stage2_ifeval_qwen3_8b_dev_variants
```

Select one variant per condition:

```bash
python -m emotion_stage2.select_prompt_variants \
  --dev-run runs/stage2_ifeval_qwen3_8b_dev_variants \
  --output runs/prompt_locks/ifeval_qwen3_8b.json
```

Run the held-out test split with that immutable lock:

```bash
python -m emotion_stage2.run_stage2 \
  --task ifeval \
  --model Qwen/Qwen3-8B \
  --split test \
  --prompt-lock runs/prompt_locks/ifeval_qwen3_8b.json \
  --output-dir runs/stage2_ifeval_qwen3_8b_test_locked
```

The runner rejects non-fixture test runs without a matching task/model lock.
Version 2 locks also freeze the complete task configuration, including the
RULER context, depth, mode, distractor, and multi-key grid. Version 1 locks
remain readable for prior runs.

## RULER

The same protocol applies with `--task ruler_niah`. Defaults are:

```text
--ruler-context-lengths 4096 8192 16384
--ruler-depths 0.1 0.5 0.9
--ruler-modes single multi
--ruler-samples-per-cell 4
```

The dev split contains one deterministic sample per grid cell. The test split
contains four disjoint samples per grid cell.

### Hardened 128k calibration

The revised Stage 2 calibration uses YaRN because Qwen3's native configured
window is 40,960 tokens. This is an explicit experimental confound and is
recorded in `metadata.json`. Calibrate task difficulty only on the two neutral
conditions:

```bash
python -m emotion_stage2.run_stage2 \
  --task ruler_niah \
  --model Qwen/Qwen3-8B \
  --split dev \
  --conditions neutral_empty neutral_length_matched \
  --ruler-hard \
  --ruler-context-lengths 32768 65536 131072 \
  --ruler-depths 0.1 0.3 0.5 0.7 0.9 \
  --ruler-modes single multi aggregate \
  --ruler-distractors 3 \
  --ruler-multi-key-count 5 \
  --ruler-aggregate-match-count 2 \
  --ruler-context-reserve-tokens 1024 \
  --ruler-samples-per-cell 3 \
  --max-model-len 131072 \
  --enable-yarn \
  --yarn-original-max-position-embeddings 40960 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --batch-size 1 \
  --output-dir runs/stage2_ruler_niah_hard_qwen3_8b_calibration
```

With no explicit `--yarn-factor`, the runner derives `131072 / 40960 = 3.2`.
It renders and tokenizes every prompt before constructing vLLM, then rejects a
run when the longest prompt plus generation allowance exceeds
`--max-model-len`.

Hard RULER runs use a 128-token generation cap so a correctly formatted
five-to-eight-code answer cannot be truncated. Aggregate match count is
controlled separately from multi-key count and is included in the task lock.

After selecting a neutral configuration with 0.5-0.8 accuracy and a visible
depth gradient, run all prompt variants on dev, create a version 2 prompt lock,
and use the same arguments for test. The validated A100-40GB configuration for
both Qwen3-8B and Qwen3-14B at 128k is `--tensor-parallel-size 4 --batch-size
4`; use the same topology for the locked dev and test runs.

## Validation

Validate prompts or a completed run:

```bash
python -m emotion_stage2.validate_run \
  --run-dir runs/stage2_ifeval_qwen3_8b_test_locked
```

Completed runs contain:

- `prompts.jsonl`
- `predictions.jsonl`
- `summary.csv`
- `paired_deltas.csv`
- `confusions.jsonl`
- `metadata.json`
- `run_stats.json`
- `requirements.freeze.txt`
- `nvidia-smi.txt`
- `validation.json` after validation

`summary.csv` reports per-condition bootstrap intervals for every task signal.
`paired_deltas.csv` reports paired deltas against both neutral baselines,
win/tie/loss counts, exact paired sign-test p-values, and Holm-adjusted p-values
for the declared primary signal family.
