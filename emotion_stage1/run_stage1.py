"""Run the Stage 1 prompt-only affect pilot on BANKING77."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import time

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from .affect import CONDITIONS, DEFAULT_CONDITIONS
from .banking77 import (
    build_prompt,
    load_banking77_examples,
    load_fixture_examples,
    parse_prediction,
)
from .generation import GenerationConfig, make_generator, render_chat_prompts
from .metrics import confusion_counts, paired_accuracy_deltas, summarize


DEFAULT_MODEL = "Qwen/Qwen3-8B"
BASELINE_CONDITIONS = ("neutral_empty", "neutral_length_matched")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--engine", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt-format", choices=("chat", "plain"), default="chat")
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For Qwen3-style models, request non-thinking mode for parseable labels.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help=f"Affect conditions. Known: {', '.join(sorted(CONDITIONS))}",
    )
    parser.add_argument("--output-dir", default="runs/stage1_banking77")
    parser.add_argument(
        "--dry-run-prompts",
        action="store_true",
        help="Write prompts and metadata, but do not load a model.",
    )
    parser.add_argument(
        "--fixture-smoke-test",
        action="store_true",
        help="Use a tiny built-in BANKING77-like fixture instead of Hugging Face datasets.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "n",
                "accuracy",
                "parse_rate",
                "macro_f1",
                "accuracy_ci_low",
                "accuracy_ci_high",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_paired_deltas_csv(path: Path, rows) -> None:
    fieldnames = [
        "condition",
        "baseline",
        "n",
        "accuracy_delta",
        "accuracy_delta_ci_low",
        "accuracy_delta_ci_high",
        "wins",
        "ties",
        "losses",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unknown_conditions = sorted(set(args.conditions) - set(CONDITIONS))
    if unknown_conditions:
        raise SystemExit(f"Unknown conditions: {', '.join(unknown_conditions)}")

    if args.fixture_smoke_test:
        examples, labels = load_fixture_examples()
    else:
        examples, labels = load_banking77_examples(
            split=args.split,
            limit=args.limit,
            seed=args.seed,
        )

    prompt_rows = []
    prompts = []
    for condition in args.conditions:
        for example in examples:
            prompt = build_prompt(example, labels, condition)
            prompt_rows.append(
                {
                    "condition": condition,
                    "item_id": example.item_id,
                    "text": example.text,
                    "gold_label": example.gold_label,
                    "prompt": prompt,
                }
            )
            prompts.append(prompt)

    write_jsonl(output_dir / "prompts.jsonl", prompt_rows)
    metadata = {
        "model": args.model,
        "engine": args.engine,
        "dataset": "PolyAI/banking77",
        "fixture_smoke_test": args.fixture_smoke_test,
        "split": args.split,
        "limit": args.limit,
        "num_examples": len(examples),
        "num_prompts": len(prompt_rows),
        "seed": args.seed,
        "conditions": args.conditions,
        "paired_baselines": list(BASELINE_CONDITIONS),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "prompt_format": args.prompt_format,
        "disable_thinking": args.disable_thinking,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.dry_run_prompts:
        print(f"Wrote {len(prompt_rows)} prompts to {output_dir / 'prompts.jsonl'}")
        return

    generator = make_generator(
        args.engine,
        GenerationConfig(
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            dtype=args.dtype,
        ),
    )

    start_time = time()
    model_prompts = prompts
    if args.prompt_format == "chat":
        model_prompts = render_chat_prompts(
            prompts,
            model=args.model,
            disable_thinking=args.disable_thinking,
        )

    raw_outputs = []
    for start in tqdm(range(0, len(model_prompts), args.batch_size), desc="generating"):
        batch = model_prompts[start : start + args.batch_size]
        raw_outputs.extend(generator.generate(batch))
    elapsed_seconds = time() - start_time

    rows = []
    for prompt_row, raw_output in zip(prompt_rows, raw_outputs, strict=True):
        pred_label, parse_method = parse_prediction(raw_output, labels)
        rows.append(
            {
                "condition": prompt_row["condition"],
                "item_id": prompt_row["item_id"],
                "text": prompt_row["text"],
                "gold_label": prompt_row["gold_label"],
                "raw_output": raw_output,
                "pred_label": pred_label,
                "parse_method": parse_method,
                "correct": pred_label == prompt_row["gold_label"],
            }
        )

    write_jsonl(output_dir / "predictions.jsonl", rows)
    summary_rows = summarize(rows, seed=args.seed, labels=labels)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    paired_rows = paired_accuracy_deltas(
        rows,
        baselines=BASELINE_CONDITIONS,
        seed=args.seed,
    )
    write_paired_deltas_csv(output_dir / "paired_deltas.csv", paired_rows)
    write_jsonl(output_dir / "confusions.jsonl", confusion_counts(rows))

    run_stats = {
        "elapsed_seconds": elapsed_seconds,
        "num_generations": len(rows),
        "generations_per_second": len(rows) / elapsed_seconds if elapsed_seconds else None,
    }
    (output_dir / "run_stats.json").write_text(json.dumps(run_stats, indent=2), encoding="utf-8")

    print(f"Wrote predictions to {output_dir / 'predictions.jsonl'}")
    print(f"Wrote summary to {output_dir / 'summary.csv'}")
    print(f"Wrote paired deltas to {output_dir / 'paired_deltas.csv'}")


if __name__ == "__main__":
    main()
