"""Evaluate held-out Stage 3 probes and enforce the pre-steering gate."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from emotion_stage1.affect import EMOTION_VECTOR_CONCEPTS

from .io import read_jsonl, write_csv
from .vectors import (
    COMBINED_PAIR_NAME,
    CONCEPT_PAIRS,
    PAIR_NAMES,
    POSITION_POLICIES,
)


_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

AFFECT_LEXICON: dict[str, tuple[str, ...]] = {
    "calm": (
        "relaxed",
        "steady",
        "unhurried",
        "serene",
        "composed",
        "ease",
        "quiet",
        "assured",
        "assurance",
        "peaceful",
        "rush",
        "pressure",
        "hurried",
        "panic",
        "frantic",
    ),
    "desperate": (
        "frantic",
        "panic",
        "panicked",
        "dread",
        "anxious",
        "anxiety",
        "urgent",
        "urgency",
        "overwhelmed",
        "hopeless",
        "despair",
    ),
    "focused": (
        "attentive",
        "attention",
        "concentrated",
        "concentration",
        "absorbed",
        "precise",
        "precision",
        "deliberate",
        "sharp",
        "focus",
    ),
    "distracted": (
        "wandering",
        "scattered",
        "unfocused",
        "restless",
        "drifting",
        "distraction",
    ),
    "patient": (
        "unhurried",
        "tolerant",
        "measured",
        "waiting",
        "waited",
        "waits",
        "gentle",
        "patiently",
    ),
    "rushed": (
        "hurried",
        "hasty",
        "frantic",
        "scrambling",
        "pressured",
        "quick",
        "quickly",
        "rushing",
        "rush",
    ),
    "responsible": (
        "diligent",
        "diligence",
        "conscientious",
        "careful",
        "carefully",
        "accountable",
        "thorough",
        "thoroughly",
        "dutiful",
        "responsibility",
    ),
    "careless": (
        "sloppy",
        "negligent",
        "reckless",
        "hasty",
        "indifferent",
        "carelessly",
        "carelessness",
    ),
    "curious": (
        "intrigued",
        "eager",
        "interested",
        "inquisitive",
        "wondering",
        "curiosity",
    ),
    "bored": (
        "dull",
        "tedious",
        "monotonous",
        "monotony",
        "listless",
        "weary",
        "uninterested",
        "boredom",
    ),
}
AFFECT_MASK = set(EMOTION_VECTOR_CONCEPTS).union(
    *AFFECT_LEXICON.values()
)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute tie-aware binary ROC-AUC without scipy/sklearn."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    positive_rank_sum = ranks[labels].sum()
    return float(
        (
            positive_rank_sum
            - positive_count * (positive_count + 1) / 2.0
        )
        / (positive_count * negative_count)
    )


def balanced_accuracy(
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
) -> float:
    positive_mean = float(train_scores[train_labels].mean())
    negative_mean = float(train_scores[~train_labels].mean())
    threshold = (positive_mean + negative_mean) / 2.0
    predictions = validation_scores >= threshold
    if positive_mean < negative_mean:
        predictions = ~predictions
    true_positive_rate = predictions[validation_labels].mean()
    true_negative_rate = (~predictions[~validation_labels]).mean()
    return float((true_positive_rate + true_negative_rate) / 2.0)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in _WORD_RE.findall(text.lower())
        if token not in AFFECT_MASK
    ]


def lexical_scores(
    train_texts: list[str],
    train_labels: np.ndarray,
    validation_texts: list[str],
) -> np.ndarray:
    """Naive-Bayes bag-of-words control with affect vocabulary masked."""

    positive = Counter()
    negative = Counter()
    for text, label in zip(train_texts, train_labels, strict=True):
        (positive if label else negative).update(_tokens(text))
    vocabulary = sorted(set(positive) | set(negative))
    vocabulary_size = max(1, len(vocabulary))
    positive_total = sum(positive.values()) + vocabulary_size
    negative_total = sum(negative.values()) + vocabulary_size
    log_odds = {
        token: math.log((positive[token] + 1) / positive_total)
        - math.log((negative[token] + 1) / negative_total)
        for token in vocabulary
    }
    return np.asarray(
        [
            sum(log_odds.get(token, 0.0) for token in _tokens(text))
            for text in validation_texts
        ],
        dtype=np.float64,
    )


def _pair_labels(emotion_indices: np.ndarray, pair_name: str) -> tuple[np.ndarray, np.ndarray]:
    if pair_name == COMBINED_PAIR_NAME:
        positive_ids = {
            EMOTION_VECTOR_CONCEPTS.index(positive)
            for positive, _ in CONCEPT_PAIRS
        }
        negative_ids = {
            EMOTION_VECTOR_CONCEPTS.index(negative)
            for _, negative in CONCEPT_PAIRS
        }
    else:
        pair_index = PAIR_NAMES.index(pair_name)
        positive, negative = CONCEPT_PAIRS[pair_index]
        positive_ids = {EMOTION_VECTOR_CONCEPTS.index(positive)}
        negative_ids = {EMOTION_VECTOR_CONCEPTS.index(negative)}
    selected = np.asarray(
        [
            index in positive_ids or index in negative_ids
            for index in emotion_indices
        ],
        dtype=bool,
    )
    labels = np.asarray(
        [index in positive_ids for index in emotion_indices[selected]],
        dtype=bool,
    )
    return selected, labels


def _fit_logistic_probe_auc(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
    steps: int,
) -> float:
    import torch

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    validation = torch.as_tensor(
        validation_x,
        dtype=torch.float32,
        device=device,
    )
    center = train.mean(dim=0, keepdim=True)
    scale = train.std(dim=0, keepdim=True).clamp_min(1e-4)
    train = (train - center) / scale
    validation = (validation - center) / scale
    targets = torch.as_tensor(
        train_y.astype(np.float32),
        device=device,
    ).unsqueeze(1)
    probe = torch.nn.Linear(train.shape[1], 1, device=device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.03, weight_decay=1e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            probe(train),
            targets,
        )
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        scores = probe(validation).squeeze(1).cpu().numpy()
    return roc_auc(scores, validation_y)


def gate_decision(
    *,
    direction_auc: float,
    lexical_auc: float,
    shuffled_control: float,
    random_control: float,
    min_auc: float,
    gate_margin: float,
    lexical_cap: float,
) -> dict:
    lexical_unadjudicable = lexical_auc > lexical_cap
    passes_nonlexical_controls = (
        direction_auc >= min_auc
        and direction_auc >= shuffled_control + gate_margin
        and direction_auc >= random_control + gate_margin
    )
    passes_lexical_control = (
        not lexical_unadjudicable
        and direction_auc >= lexical_auc + gate_margin
    )
    return {
        "lexical_verdict": (
            "abstain" if lexical_unadjudicable else "applied"
        ),
        "passes_nonlexical_controls": passes_nonlexical_controls,
        "passes_gate": (
            passes_nonlexical_controls and passes_lexical_control
        ),
        "pass_with_lexical_abstain": (
            passes_nonlexical_controls and lexical_unadjudicable
        ),
    }


def write_layer_profile_figure(
    output_dir: Path,
    profile_rows: list[dict],
    *,
    model: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair_names = list(dict.fromkeys(row["pair"] for row in profile_rows))
    figure, axes = plt.subplots(
        1,
        len(POSITION_POLICIES),
        figsize=(13, 5),
        sharey=True,
        constrained_layout=True,
    )
    for axis, policy in zip(axes, POSITION_POLICIES, strict=True):
        policy_rows = [
            row for row in profile_rows if row["policy"] == policy
        ]
        for pair_name in pair_names:
            rows = sorted(
                (
                    row
                    for row in policy_rows
                    if row["pair"] == pair_name
                ),
                key=lambda row: int(row["layer"]),
            )
            axis.plot(
                [int(row["layer"]) for row in rows],
                [float(row["direction_auc"]) for row in rows],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=pair_name,
            )
        diligence_rows = sorted(
            (
                row
                for row in policy_rows
                if row["pair"] == COMBINED_PAIR_NAME
            ),
            key=lambda row: int(row["layer"]),
        )
        axis.plot(
            [int(row["layer"]) for row in diligence_rows],
            [float(row["lexical_auc"]) for row in diligence_rows],
            color="black",
            linestyle="--",
            linewidth=1.25,
            label="diligence lexical control",
        )
        axis.axhline(
            float(policy_rows[0]["min_auc"]),
            color="gray",
            linestyle=":",
            linewidth=1,
            label="minimum AUC",
        )
        axis.set_title(f"{policy}-pooled activations")
        axis.set_xlabel("Decoder layer")
        axis.set_ylim(0.45, 1.02)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out ROC-AUC")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        fontsize=8,
    )
    figure.suptitle(f"Stage 3 layer profile: {model}")
    figure.savefig(output_dir / "layer_profile.png", dpi=180)
    figure.savefig(output_dir / "layer_profile.svg")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--vignettes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-auc", type=float, default=0.75)
    parser.add_argument("--gate-margin", type=float, default=0.02)
    parser.add_argument("--lexical-cap", type=float, default=0.90)
    parser.add_argument("--logistic-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--stage", choices=("3", "3.5"), default="3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.output_dir / "probe_report.json"
    if report_path.exists():
        raise SystemExit(f"Refusing to overwrite {report_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(
        (args.vector_dir / "extraction_meta.json").read_text(encoding="utf-8")
    )
    vectors = np.load(args.vector_dir / "vectors.npz")
    activations = np.load(args.vector_dir / "activations.npz")
    vignette_rows = read_jsonl(args.vignettes)
    by_id = {int(row["vignette_id"]): row for row in vignette_rows}
    ordered_rows = [by_id[int(value)] for value in activations["vignette_ids"]]

    layers = vectors["layers"].astype(int)
    pair_names = [str(value) for value in vectors["pair_names"]]
    emotion_indices = activations["emotion_indices"].astype(int)
    replicates = activations["replicates"].astype(int)
    train_replicate_count = len(meta["train_replicates"])
    train_mask_all = replicates < train_replicate_count
    validation_mask_all = ~train_mask_all
    texts = np.asarray([row["text"] for row in ordered_rows], dtype=object)

    profile_rows = []
    best_rows = []
    for pair_index, pair_name in enumerate(pair_names):
        pair_selected, pair_labels = _pair_labels(emotion_indices, pair_name)
        pair_train = train_mask_all[pair_selected]
        pair_validation = validation_mask_all[pair_selected]
        train_labels = pair_labels[pair_train]
        validation_labels = pair_labels[pair_validation]
        pair_texts = texts[pair_selected]
        lexical = lexical_scores(
            list(pair_texts[pair_train]),
            train_labels,
            list(pair_texts[pair_validation]),
        )
        lexical_auc = roc_auc(lexical, validation_labels)

        for policy in POSITION_POLICIES:
            policy_activations = activations[policy][pair_selected]
            units = vectors[f"{policy}_unit"][pair_index]
            shuffled_units = vectors[f"{policy}_shuffled_unit"][pair_index]
            random_units = vectors[f"{policy}_random_unit"][pair_index]
            policy_rows = []
            for layer_offset, layer_index in enumerate(layers):
                train_x = policy_activations[pair_train, layer_offset]
                validation_x = policy_activations[pair_validation, layer_offset]
                direction_train = train_x @ units[layer_offset]
                direction_validation = validation_x @ units[layer_offset]
                shuffled_validation = (
                    validation_x @ shuffled_units[layer_offset]
                )
                random_validation = validation_x @ random_units[layer_offset]
                direction_auc = roc_auc(direction_validation, validation_labels)
                shuffled_auc = roc_auc(shuffled_validation, validation_labels)
                random_auc = roc_auc(random_validation, validation_labels)
                shuffled_control = max(shuffled_auc, 1.0 - shuffled_auc)
                random_control = max(random_auc, 1.0 - random_auc)
                decision = gate_decision(
                    direction_auc=direction_auc,
                    lexical_auc=lexical_auc,
                    shuffled_control=shuffled_control,
                    random_control=random_control,
                    min_auc=args.min_auc,
                    gate_margin=args.gate_margin,
                    lexical_cap=args.lexical_cap,
                )
                row = {
                    "pair": pair_name,
                    "policy": policy,
                    "layer": int(layer_index),
                    "n_train": int(pair_train.sum()),
                    "n_validation": int(pair_validation.sum()),
                    "direction_auc": direction_auc,
                    "balanced_accuracy": balanced_accuracy(
                        direction_train,
                        train_labels,
                        direction_validation,
                        validation_labels,
                    ),
                    "lexical_auc": lexical_auc,
                    "shuffled_auc_oriented": shuffled_control,
                    "random_auc_oriented": random_control,
                    "min_auc": args.min_auc,
                    "gate_margin": args.gate_margin,
                    "lexical_cap": args.lexical_cap,
                    **decision,
                }
                profile_rows.append(row)
                policy_rows.append((row, train_x, validation_x))
            best, train_x, validation_x = max(
                policy_rows,
                key=lambda item: item[0]["direction_auc"],
            )
            best = dict(best)
            best["logistic_probe_auc"] = _fit_logistic_probe_auc(
                train_x,
                train_labels,
                validation_x,
                validation_labels,
                seed=args.seed + pair_index,
                steps=args.logistic_steps,
            )
            best_rows.append(best)

    selected_lexical_applied = []
    selected_with_abstain = []
    for pair_name in pair_names:
        candidates = [row for row in best_rows if row["pair"] == pair_name]
        best = max(candidates, key=lambda row: row["direction_auc"])
        if best["passes_gate"]:
            selected_lexical_applied.append(best)
        elif best["pass_with_lexical_abstain"]:
            selected_with_abstain.append(best)
    report = {
        "stage": 3 if args.stage == "3" else 3.5,
        "artifact": "probe_gate",
        "model": meta["model"],
        "vector_dir": str(args.vector_dir.resolve()),
        "vignettes": str(args.vignettes.resolve()),
        "split": {
            "train_replicates": meta["train_replicates"],
            "validation_replicates": meta["validation_replicates"],
        },
        "min_auc": args.min_auc,
        "gate_margin": args.gate_margin,
        "lexical_cap": args.lexical_cap,
        "gate_passed": bool(selected_lexical_applied),
        "gate_passed_with_abstain": bool(
            selected_lexical_applied or selected_with_abstain
        ),
        "passing_directions": selected_lexical_applied,
        "passing_directions_lexical_applied": selected_lexical_applied,
        "passing_directions_with_abstain": selected_with_abstain,
        "primary_diligence_passed": any(
            row["pair"] == COMBINED_PAIR_NAME
            for row in selected_lexical_applied
        ),
        "primary_diligence_passed_with_abstain": any(
            row["pair"] == COMBINED_PAIR_NAME
            for row in (
                selected_lexical_applied + selected_with_abstain
            )
        ),
        "best_by_pair_and_policy": best_rows,
        "gate_rule": (
            "clean pass requires direction_auc >= min_auc and exceeds "
            "masked bag-of-words, shuffled-label, and random-vector AUC by "
            "gate_margin; lexical_auc > lexical_cap abstains and can only "
            "produce a flagged nonlexical-controls pass"
        ),
    }
    write_csv(args.output_dir / "layer_profile.csv", profile_rows)
    write_layer_profile_figure(
        args.output_dir,
        profile_rows,
        model=meta["model"],
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "gate.json").write_text(
        json.dumps(
            {
                "gate_passed": report["gate_passed"],
                "gate_passed_with_abstain": report[
                    "gate_passed_with_abstain"
                ],
                "primary_diligence_passed": report[
                    "primary_diligence_passed"
                ],
                "primary_diligence_passed_with_abstain": report[
                    "primary_diligence_passed_with_abstain"
                ],
                "lexical_cap": args.lexical_cap,
                "passing_directions": selected_lexical_applied,
                "passing_directions_lexical_applied": (
                    selected_lexical_applied
                ),
                "passing_directions_with_abstain": selected_with_abstain,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
