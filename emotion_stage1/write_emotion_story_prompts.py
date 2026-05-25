"""Write Anthropic-style synthetic story prompts for later vector extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .affect import iter_emotion_story_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--output", default="runs/emotion_story_prompts.jsonl")
    args = parser.parse_args()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in iter_emotion_story_prompts(replicates=args.replicates):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
