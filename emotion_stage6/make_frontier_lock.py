"""Write the availability-locked Stage 6 frontier model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFERRED_PROFILES = (
    "us.anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=PREFERRED_PROFILES, required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--smoke-output", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    lock = {
        "version": 1,
        "stage": 6,
        "artifact": "frontier_model_lock",
        "provider": "Amazon Bedrock",
        "model_family": "Anthropic Claude Sonnet",
        "model_id": args.model_id,
        "region": args.region,
        "selection_order": list(PREFERRED_PROFILES),
        "selection_rule": (
            "first active profile in the preregistered order that returns the "
            "exact requested label in a one-item format smoke; no outcome-based "
            "switching"
        ),
        "smoke_output": args.smoke_output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
