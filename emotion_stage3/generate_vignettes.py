"""Generate the matched Stage 3 affect-vignette corpus with vLLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import time

from emotion_stage1.affect import iter_emotion_story_prompts
from emotion_stage1.generation import (
    GenerationConfig,
    VLLMGenerator,
    render_chat_prompts,
)

from .io import write_environment_snapshots, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--strict-stimuli", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise SystemExit("--replicates must be positive")
    output = args.output_dir / "vignettes.jsonl"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(
        iter_emotion_story_prompts(
            replicates=args.replicates,
            strict_stimuli=args.strict_stimuli,
        )
    )
    if args.limit is not None:
        rows = rows[: args.limit]
    prompts = render_chat_prompts(
        [row["prompt"] for row in rows],
        model=args.model,
        disable_thinking=True,
    )
    generator = VLLMGenerator(
        GenerationConfig(
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    )
    started = time()
    stories = generator.generate(prompts)
    elapsed = time() - started
    if len(stories) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} stories, received {len(stories)}")

    output_rows = []
    for index, (row, story) in enumerate(zip(rows, stories, strict=True)):
        text = story.strip()
        output_rows.append(
            {
                "vignette_id": index,
                **row,
                "text": text,
                "word_count": len(text.split()),
                "names_target_concept": bool(
                    re.search(
                        rf"\b{re.escape(row['emotion'])}\b",
                        text,
                        flags=re.IGNORECASE,
                    )
                ),
            }
        )
    write_jsonl(output, output_rows)
    metadata = {
        "stage": 3,
        "artifact": "vignettes",
        "model": args.model,
        "replicates": args.replicates,
        "num_vignettes": len(output_rows),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "strict_stimuli": args.strict_stimuli,
        "stimulus_design": (
            "matched_contrastive_pairs"
            if args.strict_stimuli
            else "independent_concept_prompts"
        ),
        "num_matched_pairs": len(
            {
                row["pair_id"]
                for row in output_rows
                if "pair_id" in row
            }
        ),
        "elapsed_seconds": elapsed,
        "named_target_concept_count": sum(
            row["names_target_concept"] for row in output_rows
        ),
        "word_count_min": min((row["word_count"] for row in output_rows), default=0),
        "word_count_max": max((row["word_count"] for row in output_rows), default=0),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_environment_snapshots(args.output_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
