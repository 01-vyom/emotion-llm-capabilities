"""Text-only baselines for the probe stimuli: does word ORDER alone separate the poles?

Trains an L2 logistic regression on unigram counts and on bigram counts of the raw story
text, using the same split as the activation probe (replicates 0-13 train, 14-19 held out),
per emotion pair and pooled ("diligence" composite). Reports held-out AUC.

Usage:  python docs/figures/bigram_baseline.py path/to/vignettes.jsonl
Writes docs/figures/data/bigram_baseline.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "bigram_baseline.csv"
TRAIN_REPLICATES = 14


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def ngrams(toks: list[str], n: int) -> list[str]:
    return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def featurize(texts: list[str], n: int, vocab: dict[str, int] | None = None):
    grams = [Counter(ngrams(tokens(t), n)) for t in texts]
    if vocab is None:
        vocab = {g: i for i, g in enumerate(sorted({g for c in grams for g in c}))}
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row, c in enumerate(grams):
        for g, k in c.items():
            if g in vocab:
                x[row, vocab[g]] = k
    return x, vocab


def fit_logreg(x: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 3000, lr: float = 0.5):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(xs @ w + b)))
        g = p - y
        w -= lr * (xs.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return lambda z: ((z - mu) / sd) @ w + b


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    for s in np.unique(scores):
        m = scores == s
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    pos = labels.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(path: str) -> None:
    rows = [json.loads(line) for line in open(path)]
    pairs = sorted({r["pair_name"] for r in rows})
    results = []
    for name in pairs + ["diligence (all pairs pooled)"]:
        sub = rows if name.startswith("diligence") else [r for r in rows if r["pair_name"] == name]
        tr = [r for r in sub if r["replicate"] < TRAIN_REPLICATES]
        te = [r for r in sub if r["replicate"] >= TRAIN_REPLICATES]
        ytr = np.array([r["pole"] == "positive" for r in tr], dtype=float)
        yte = np.array([r["pole"] == "positive" for r in te], dtype=float)
        row = {"pair": name, "n_train": len(tr), "n_heldout": len(te)}
        for n in (1, 2):
            xtr, vocab = featurize([r["text"] for r in tr], n)
            xte, _ = featurize([r["text"] for r in te], n, vocab)
            score = fit_logreg(xtr, ytr)
            row[f"{'unigram' if n == 1 else 'bigram'}_auc"] = round(auc(score(xte), yte), 4)
        results.append(row)
        print(row)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
