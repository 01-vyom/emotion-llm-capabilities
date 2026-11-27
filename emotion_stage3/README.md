# Stage 3

Stage 3 extracts contrastive affect/diligence directions from Qwen3 residual
activations, validates them on held-out vignettes, and gates activation steering
before any task claims are made. The execution contract is
`docs/stage3-plan.md`.

## Artifact flow

Generate 20 matched replicates per concept:

```bash
python -m emotion_stage3.generate_vignettes \
  --model Qwen/Qwen3-8B \
  --replicates 20 \
  --strict-stimuli \
  --output-dir runs/stage3_qwen3_8b_vignettes
```

Extract fp32 vectors and held-out activations. Replicates 0-13 train each
direction; replicates 14-19 are never used to construct it:

```bash
python -m emotion_stage3.vectors \
  --model Qwen/Qwen3-8B \
  --vignettes runs/stage3_qwen3_8b_vignettes/vignettes.jsonl \
  --train-replicates 14 \
  --output-dir runs/stage3_qwen3_8b_vectors
```

Apply the leakage-controlled probe gate:

```bash
python -m emotion_stage3.probe \
  --vector-dir runs/stage3_qwen3_8b_vectors \
  --vignettes runs/stage3_qwen3_8b_vignettes/vignettes.jsonl \
  --output-dir runs/stage3_qwen3_8b_probe
```

The gate requires held-out activation AUC to be at least 0.75 and to exceed the
masked bag-of-words, shuffled-label, and random-vector controls by 0.02. If no
direction passes with the lexical control applied, Stage 3 stops before steering
as specified in the plan. A lexical AUC above 0.90 abstains rather than vetoes;
abstain-only directions are reported separately and do not authorize
confirmatory steering.
The probe artifact includes CSV, PNG, and SVG layer profiles.

Backend parity between matched unsteered Transformers and vLLM runs is recorded
with:

```bash
python -m emotion_stage3.compare_backends \
  --transformers-run runs/stage3_backend_parity_transformers \
  --vllm-run runs/stage3_backend_parity_vllm \
  --output-dir runs/stage3_backend_parity_report
```

Set `NLTK_DATA` to a directory containing both `punkt` and `punkt_tab` when
running IFEval with NLTK 3.9 or newer.

## Steering

Calibration is dev-only. Alpha is a fraction of the typical training-token
residual norm, so `0.05` injects a unit direction with norm equal to 5% of the
layer's typical residual norm:

```bash
python -m emotion_stage3.run_stage3 \
  --task ifeval \
  --model Qwen/Qwen3-8B \
  --design calibration \
  --split dev \
  --vector-dir runs/stage3_qwen3_8b_vectors \
  --direction diligence \
  --policy mean \
  --layers 19 \
  --calibration-alphas 0.01 0.02 0.05 0.1 \
  --output-dir runs/stage3_ifeval_qwen3_8b_calibration_l19
```

`emotion_stage3.select_steering` writes a versioned lock from one or more dev
calibration runs. Held-out confirmatory runs reject missing locks or vector-hash
drift. The confirmatory design includes no-steer, positive, negative, matched
random, shuffled-label, affect-prompt/opposite-vector, and
affect-prompt/matching-vector arms.

Validate every artifact:

```bash
python -m emotion_stage3.validate_run --vector-dir RUN_DIR
python -m emotion_stage3.validate_run --probe-dir RUN_DIR
python -m emotion_stage3.validate_run --run-dir RUN_DIR
```

For 32k/64k RULER, use `--attention-implementation sdpa`, batch size 1, and YaRN
with the requested maximum window. Eager attention remains the extraction and
short-context parity path.
