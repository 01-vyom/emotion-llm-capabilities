"""Extract residual activations and build Stage 3 CAA vector artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np

from emotion_stage1.affect import (
    EMOTION_CONCEPT_PAIRS,
    EMOTION_VECTOR_CONCEPTS,
)

from .io import (
    read_jsonl,
    sha256_file,
    write_environment_snapshots,
)
from .steering import SteeredGenerator, TransformersConfig

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable


CONCEPT_PAIRS = EMOTION_CONCEPT_PAIRS
PAIR_NAMES = tuple(f"{positive}_vs_{negative}" for positive, negative in CONCEPT_PAIRS)
COMBINED_PAIR_NAME = "diligence"
POSITION_POLICIES = ("mean", "mean50", "last")
MEAN_TAIL_START_TOKEN = 50


def default_layer_grid(num_hidden_layers: int, stride: int = 4) -> tuple[int, ...]:
    if num_hidden_layers <= 0 or stride <= 0:
        raise ValueError("Layer count and stride must be positive")
    layers = list(range(stride - 1, num_hidden_layers, stride))
    if num_hidden_layers - 1 not in layers:
        layers.append(num_hidden_layers - 1)
    return tuple(layers)


def _unit_vectors(
    values: np.ndarray,
    *,
    zero_fallback_rng: np.random.Generator | None = None,
) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    zero_rows = np.argwhere(norms[..., 0] == 0)
    if len(zero_rows) and zero_fallback_rng is None:
        raise ValueError("Cannot normalize a zero-length direction")
    for row in zero_rows:
        index = tuple(int(value) for value in row)
        values[index] = zero_fallback_rng.standard_normal(
            values.shape[-1],
        ).astype(np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return (values / norms).astype(np.float32)


def build_vector_arrays(
    activations: np.ndarray,
    emotion_indices: np.ndarray,
    replicates: np.ndarray,
    *,
    train_replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Build real, shuffled, random, and combined directions for one policy."""

    train_mask = replicates < train_replicates
    rng = np.random.default_rng(seed)
    raw_directions = []
    shuffled_directions = []
    for positive, negative in CONCEPT_PAIRS:
        positive_id = EMOTION_VECTOR_CONCEPTS.index(positive)
        negative_id = EMOTION_VECTOR_CONCEPTS.index(negative)
        positive_rows = activations[train_mask & (emotion_indices == positive_id)]
        negative_rows = activations[train_mask & (emotion_indices == negative_id)]
        if len(positive_rows) == 0 or len(negative_rows) == 0:
            raise ValueError(f"Missing training rows for {positive} vs {negative}")
        raw_directions.append(
            positive_rows.mean(axis=0) - negative_rows.mean(axis=0)
        )

        paired_rows = np.concatenate([positive_rows, negative_rows], axis=0)
        labels = np.concatenate(
            [
                np.ones(len(positive_rows), dtype=bool),
                np.zeros(len(negative_rows), dtype=bool),
            ]
        )
        shuffled_labels = rng.permutation(labels)
        shuffled_directions.append(
            paired_rows[shuffled_labels].mean(axis=0)
            - paired_rows[~shuffled_labels].mean(axis=0)
        )

    raw = np.asarray(raw_directions, dtype=np.float32)
    shuffled = np.asarray(shuffled_directions, dtype=np.float32)
    raw = np.concatenate([raw, raw.mean(axis=0, keepdims=True)], axis=0)
    shuffled = np.concatenate(
        [shuffled, shuffled.mean(axis=0, keepdims=True)],
        axis=0,
    )
    random = rng.standard_normal(raw.shape, dtype=np.float32)
    return {
        "raw": raw,
        "unit": _unit_vectors(raw),
        "shuffled_unit": _unit_vectors(
            shuffled,
            zero_fallback_rng=rng,
        ),
        "random_unit": _unit_vectors(random),
    }


def _pool_hidden(hidden, attention_mask):
    import torch

    mask = attention_mask.to(hidden.device, dtype=torch.float32)
    float_hidden = hidden.float()
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (float_hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator
    token_positions = attention_mask.to(hidden.device).cumsum(dim=1) - 1
    tail_mask = mask * (token_positions >= MEAN_TAIL_START_TOKEN)
    empty_tail = tail_mask.sum(dim=1) == 0
    if empty_tail.any():
        tail_mask[empty_tail] = mask[empty_tail]
    tail_denominator = tail_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean50 = (
        float_hidden * tail_mask.unsqueeze(-1)
    ).sum(dim=1) / tail_denominator
    last_indices = attention_mask.sum(dim=1).to(hidden.device) - 1
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    last = float_hidden[batch_indices, last_indices]
    token_norm = (
        torch.linalg.vector_norm(float_hidden, dim=-1) * mask
    ).sum(dim=1) / denominator.squeeze(1)
    return mean, mean50, last, token_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vignettes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--train-replicates", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--device-map")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stage", choices=("3", "3.5"), default="3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vector_path = args.output_dir / "vectors.npz"
    if vector_path.exists():
        raise SystemExit(f"Refusing to overwrite {vector_path}")
    rows = read_jsonl(args.vignettes)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No vignettes found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generator = SteeredGenerator(
        TransformersConfig(
            model=args.model,
            max_new_tokens=1,
            batch_size=args.batch_size,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            device_map=args.device_map,
        )
    )
    model = generator.model
    tokenizer = generator.tokenizer
    tokenizer.padding_side = "right"
    num_hidden_layers = int(model.config.num_hidden_layers)
    layers = tuple(
        args.layers
        if args.layers is not None
        else default_layer_grid(num_hidden_layers, args.layer_stride)
    )
    if len(set(layers)) != len(layers):
        raise SystemExit("Duplicate layer indices are not allowed")
    if any(layer < 0 or layer >= num_hidden_layers for layer in layers):
        raise SystemExit(
            f"Layers must be within [0, {num_hidden_layers - 1}]: {layers}"
        )

    encoded_lengths = [
        len(tokenizer.encode(row["text"], add_special_tokens=True))
        for row in rows
    ]
    longest = max(encoded_lengths)
    if longest > args.max_length:
        raise SystemExit(
            f"Vignette has {longest} tokens, exceeding --max-length "
            f"{args.max_length}; refusing silent truncation"
        )

    n = len(rows)
    hidden_size = int(model.config.hidden_size)
    shape = (n, len(layers), hidden_size)
    mean_activations = np.empty(shape, dtype=np.float32)
    mean50_activations = np.empty(shape, dtype=np.float32)
    last_activations = np.empty(shape, dtype=np.float32)
    residual_norms = np.empty((n, len(layers)), dtype=np.float32)

    started = time()
    for start in tqdm(
        range(0, n, args.batch_size),
        desc="extracting activations",
    ):
        batch_rows = rows[start : start + args.batch_size]
        inputs = tokenizer(
            [row["text"] for row in batch_rows],
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        inputs = {
            key: value.to(generator.input_device)
            for key, value in inputs.items()
        }
        with generator.torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        for layer_offset, layer_index in enumerate(layers):
            hidden = outputs.hidden_states[layer_index + 1]
            mean, mean50, last, token_norm = _pool_hidden(
                hidden,
                inputs["attention_mask"],
            )
            stop = start + len(batch_rows)
            mean_activations[start:stop, layer_offset] = mean.cpu().numpy()
            mean50_activations[start:stop, layer_offset] = mean50.cpu().numpy()
            last_activations[start:stop, layer_offset] = last.cpu().numpy()
            residual_norms[start:stop, layer_offset] = token_norm.cpu().numpy()
        del outputs

    emotion_indices = np.asarray(
        [EMOTION_VECTOR_CONCEPTS.index(row["emotion"]) for row in rows],
        dtype=np.int16,
    )
    topic_names = tuple(dict.fromkeys(str(row["topic"]) for row in rows))
    topic_indices = np.asarray(
        [topic_names.index(str(row["topic"])) for row in rows],
        dtype=np.int16,
    )
    replicates = np.asarray([int(row["replicate"]) for row in rows], dtype=np.int16)
    vignette_ids = np.asarray(
        [int(row["vignette_id"]) for row in rows],
        dtype=np.int32,
    )
    validation_mask = replicates >= args.train_replicates
    if not validation_mask.any():
        raise SystemExit("The extraction corpus has no held-out replicate rows")

    arrays: dict[str, np.ndarray] = {
        "layers": np.asarray(layers, dtype=np.int16),
        "pair_names": np.asarray((*PAIR_NAMES, COMBINED_PAIR_NAME)),
        "concept_names": np.asarray(EMOTION_VECTOR_CONCEPTS),
    }
    for policy, activations in (
        ("mean", mean_activations),
        ("mean50", mean50_activations),
        ("last", last_activations),
    ):
        vector_arrays = build_vector_arrays(
            activations,
            emotion_indices,
            replicates,
            train_replicates=args.train_replicates,
            seed=args.seed + (0 if policy == "mean" else 1000),
        )
        for name, values in vector_arrays.items():
            arrays[f"{policy}_{name}"] = values

    train_mask = replicates < args.train_replicates
    arrays["residual_norm"] = residual_norms[train_mask].mean(axis=0)
    np.savez_compressed(vector_path, **arrays)
    np.savez(
        args.output_dir / "activations.npz",
        mean=mean_activations,
        mean50=mean50_activations,
        last=last_activations,
        residual_norms=residual_norms,
        emotion_indices=emotion_indices,
        topic_indices=topic_indices,
        replicates=replicates,
        vignette_ids=vignette_ids,
        topic_names=np.asarray(topic_names),
    )

    elapsed = time() - started
    metadata = {
        "stage": 3 if args.stage == "3" else 3.5,
        "artifact": "activation_vectors",
        "model": args.model,
        "source_vignettes": str(args.vignettes.resolve()),
        "source_sha256": sha256_file(args.vignettes),
        "num_vignettes": n,
        "train_replicates": list(range(args.train_replicates)),
        "validation_replicates": sorted(
            {int(value) for value in replicates[validation_mask]}
        ),
        "num_train": int(train_mask.sum()),
        "num_validation": int(validation_mask.sum()),
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "layers": list(layers),
        "position_policies": list(POSITION_POLICIES),
        "mean_tail_start_token": MEAN_TAIL_START_TOKEN,
        "concept_pairs": [
            {"name": name, "positive": positive, "negative": negative}
            for name, (positive, negative) in zip(
                PAIR_NAMES,
                CONCEPT_PAIRS,
                strict=True,
            )
        ],
        "combined_direction": COMBINED_PAIR_NAME,
        "dtype": args.dtype,
        "stored_dtype": "float32",
        "attention_implementation": args.attention_implementation,
        "device_map": args.device_map,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_observed_tokens": longest,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "alpha_parameterization": (
            "injected_delta = alpha * mean_training_token_residual_norm "
            "* unit_direction"
        ),
    }
    (args.output_dir / "extraction_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_environment_snapshots(args.output_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
