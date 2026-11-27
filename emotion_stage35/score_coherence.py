"""Score conditional output perplexity for a completed Stage 3.5 run."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

from emotion_stage3.io import read_jsonl, write_csv
from emotion_stage3.run_stage3 import load_steering_spec
from emotion_stage3.steering import (
    SteeredGenerator,
    SteeringController,
    TransformersConfig,
)
from emotion_stage3.validate_run import validate_task_run


def _score_one(
    generator: SteeredGenerator,
    rendered_prompt: str,
    output: str,
    *,
    max_output_tokens: int,
) -> tuple[float, int]:
    torch = generator.torch
    prompt_ids = generator.tokenizer.encode(
        rendered_prompt,
        add_special_tokens=False,
    )
    output_ids = generator.tokenizer.encode(
        output,
        add_special_tokens=False,
    )[:max_output_tokens]
    if not output_ids:
        return float("inf"), 0
    input_ids = torch.tensor(
        [prompt_ids + output_ids],
        dtype=torch.long,
        device=generator.input_device,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    with torch.inference_mode():
        loss = generator.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        ).loss
    return float(loss), len(output_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--samples-per-arm", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_task_run(args.run_dir)
    detail_path = args.run_dir / "coherence_perplexity.csv"
    summary_path = args.run_dir / "coherence_summary.csv"
    if detail_path.exists() or summary_path.exists():
        raise SystemExit("Refusing to overwrite existing coherence artifacts")
    metadata = json.loads(
        (args.run_dir / "metadata.json").read_text(encoding="utf-8")
    )
    prompts = read_jsonl(args.run_dir / "prompts.jsonl")
    predictions = read_jsonl(args.run_dir / "predictions.jsonl")
    prompt_by_key = {
        (row["condition"], int(row["item_id"])): row
        for row in prompts
    }
    predictions_by_condition: dict[str, list[dict]] = {}
    for row in predictions:
        predictions_by_condition.setdefault(row["condition"], []).append(row)

    generator = SteeredGenerator(
        TransformersConfig(
            model=metadata["model"],
            max_new_tokens=1,
            batch_size=1,
            dtype=metadata["dtype"],
            attention_implementation=metadata["attention_implementation"],
            device_map=metadata["device_map"],
        )
    )
    detail_rows = []
    for condition in metadata["arms"]:
        condition_rows = sorted(
            predictions_by_condition[condition],
            key=lambda row: int(row["item_id"]),
        )[: args.samples_per_arm]
        first_prompt = prompt_by_key[
            (condition, int(condition_rows[0]["item_id"]))
        ]
        spec = None
        if first_prompt["vector_kind"] is not None:
            spec = load_steering_spec(
                Path(metadata["vector_dir"]),
                pair_name=metadata["direction"],
                policy=metadata["policy"],
                layers=tuple(int(value) for value in metadata["layers"]),
                vector_kind=first_prompt["vector_kind"],
                alpha=float(first_prompt["alpha"]),
            )
        context = (
            nullcontext()
            if spec is None or spec.alpha == 0
            else SteeringController(generator.model, spec)
        )
        with context:
            for prediction in condition_rows:
                key = (condition, int(prediction["item_id"]))
                raw_prompt = prompt_by_key[key]["prompt"]
                rendered = generator.render_chat([raw_prompt])[0]
                loss, token_count = _score_one(
                    generator,
                    rendered,
                    prediction["raw_output"],
                    max_output_tokens=args.max_output_tokens,
                )
                detail_rows.append(
                    {
                        "condition": condition,
                        "item_id": prediction["item_id"],
                        "alpha": prompt_by_key[key]["alpha"],
                        "vector_kind": prompt_by_key[key]["vector_kind"],
                        "scored_output_tokens": token_count,
                        "conditional_nll": loss,
                        "conditional_perplexity": (
                            math.exp(min(loss, 20.0))
                            if math.isfinite(loss)
                            else float("inf")
                        ),
                    }
                )
    write_csv(detail_path, detail_rows)
    summary_rows = []
    for condition in metadata["arms"]:
        rows = [row for row in detail_rows if row["condition"] == condition]
        finite = [
            row for row in rows if math.isfinite(row["conditional_nll"])
        ]
        summary_rows.append(
            {
                "condition": condition,
                "n": len(rows),
                "n_finite": len(finite),
                "conditional_nll_mean": (
                    sum(row["conditional_nll"] for row in finite) / len(finite)
                    if finite
                    else float("inf")
                ),
                "conditional_perplexity_mean": (
                    sum(row["conditional_perplexity"] for row in finite)
                    / len(finite)
                    if finite
                    else float("inf")
                ),
            }
        )
    write_csv(summary_path, summary_rows)
    coherence_meta = {
        "stage": 3.5,
        "run_dir": str(args.run_dir.resolve()),
        "samples_per_arm": args.samples_per_arm,
        "max_output_tokens": args.max_output_tokens,
        "scoring": "teacher_forced_conditional_nll_under_matching_steering",
    }
    (args.run_dir / "coherence_metadata.json").write_text(
        json.dumps(coherence_meta, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(coherence_meta, indent=2))


if __name__ == "__main__":
    main()
