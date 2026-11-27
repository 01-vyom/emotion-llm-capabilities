"""Compare matched Transformers and vLLM baseline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_jsonl, write_csv


SIGNALS = ("strict_prompt", "parse_success", "output_truncated", "output_tokens")


def _index_predictions(rows: list[dict], run_dir: Path) -> dict[int, dict]:
    indexed = {}
    for row in rows:
        item_id = int(row["item_id"])
        if item_id in indexed:
            raise ValueError(f"Duplicate item_id {item_id} in {run_dir}")
        indexed[item_id] = row
    return indexed


def _mean(rows: list[dict], signal: str) -> float:
    return sum(float(row[signal]) for row in rows) / len(rows)


def compare_backend_runs(
    transformers_run: Path,
    vllm_run: Path,
    *,
    strict_tolerance: float = 0.05,
    parse_tolerance: float = 0.0,
    truncation_tolerance: float = 0.0,
) -> tuple[dict, list[dict]]:
    transformers_rows = read_jsonl(transformers_run / "predictions.jsonl")
    vllm_rows = read_jsonl(vllm_run / "predictions.jsonl")
    transformers = _index_predictions(transformers_rows, transformers_run)
    vllm = _index_predictions(vllm_rows, vllm_run)
    if not transformers:
        raise ValueError("Backend parity requires at least one prediction")
    if transformers.keys() != vllm.keys():
        missing_transformers = sorted(vllm.keys() - transformers.keys())
        missing_vllm = sorted(transformers.keys() - vllm.keys())
        raise ValueError(
            "Backend item IDs differ: "
            f"missing_transformers={missing_transformers}, "
            f"missing_vllm={missing_vllm}"
        )

    item_rows = []
    for item_id in sorted(transformers):
        transformers_row = transformers[item_id]
        vllm_row = vllm[item_id]
        item_rows.append(
            {
                "item_id": item_id,
                "source_prompt_match": (
                    transformers_row["source_prompt"]
                    == vllm_row["source_prompt"]
                ),
                "raw_output_exact_match": (
                    transformers_row["raw_output"] == vllm_row["raw_output"]
                ),
                "strict_prompt_transformers": float(
                    transformers_row["strict_prompt"]
                ),
                "strict_prompt_vllm": float(vllm_row["strict_prompt"]),
                "strict_prompt_match": (
                    float(transformers_row["strict_prompt"])
                    == float(vllm_row["strict_prompt"])
                ),
                "parse_success_transformers": float(
                    transformers_row["parse_success"]
                ),
                "parse_success_vllm": float(vllm_row["parse_success"]),
                "output_truncated_transformers": float(
                    transformers_row["output_truncated"]
                ),
                "output_truncated_vllm": float(
                    vllm_row["output_truncated"]
                ),
                "output_tokens_transformers": int(
                    transformers_row["output_tokens"]
                ),
                "output_tokens_vllm": int(vllm_row["output_tokens"]),
            }
        )

    transformers_means = {
        signal: _mean(transformers_rows, signal) for signal in SIGNALS
    }
    vllm_means = {signal: _mean(vllm_rows, signal) for signal in SIGNALS}
    deltas = {
        signal: transformers_means[signal] - vllm_means[signal]
        for signal in SIGNALS
    }
    prompt_match_rate = sum(
        bool(row["source_prompt_match"]) for row in item_rows
    ) / len(item_rows)
    exact_output_rate = sum(
        bool(row["raw_output_exact_match"]) for row in item_rows
    ) / len(item_rows)
    strict_item_agreement_rate = sum(
        bool(row["strict_prompt_match"]) for row in item_rows
    ) / len(item_rows)
    parity_passed = (
        prompt_match_rate == 1.0
        and abs(deltas["strict_prompt"]) <= strict_tolerance
        and abs(deltas["parse_success"]) <= parse_tolerance
        and abs(deltas["output_truncated"]) <= truncation_tolerance
    )
    report = {
        "stage": 3,
        "artifact": "backend_parity",
        "transformers_run": str(transformers_run.resolve()),
        "vllm_run": str(vllm_run.resolve()),
        "num_matched_items": len(item_rows),
        "prompt_match_rate": prompt_match_rate,
        "exact_output_rate": exact_output_rate,
        "strict_item_agreement_rate": strict_item_agreement_rate,
        "transformers_means": transformers_means,
        "vllm_means": vllm_means,
        "transformers_minus_vllm": deltas,
        "tolerances": {
            "strict_prompt": strict_tolerance,
            "parse_success": parse_tolerance,
            "output_truncated": truncation_tolerance,
        },
        "parity_passed": parity_passed,
        "parity_rule": (
            "matched source prompts and mean deltas within the declared "
            "strict-prompt, parse-success, and truncation tolerances"
        ),
    }
    return report, item_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformers-run", type=Path, required=True)
    parser.add_argument("--vllm-run", type=Path, required=True)
    parser.add_argument("--strict-tolerance", type=float, default=0.05)
    parser.add_argument("--parse-tolerance", type=float, default=0.0)
    parser.add_argument("--truncation-tolerance", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {args.output_dir}")
    report, item_rows = compare_backend_runs(
        args.transformers_run,
        args.vllm_run,
        strict_tolerance=args.strict_tolerance,
        parse_tolerance=args.parse_tolerance,
        truncation_tolerance=args.truncation_tolerance,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "backend_parity.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "backend_parity_items.csv", item_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
