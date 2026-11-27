"""Bedrock helpers for the Stage 6 frontier prompt-only point."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrontierResult:
    raw_output: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int | None


class BedrockGenerator:
    def __init__(
        self,
        *,
        model_id: str,
        region: str,
        max_tokens: int,
        workers: int,
    ):
        import boto3
        from botocore.config import Config

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.workers = workers
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=max(16, workers + 4),
                connect_timeout=30,
                read_timeout=180,
            ),
        )

    def _generate_one(self, prompt: str) -> FrontierResult:
        response = self.client.converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        content = response["output"]["message"]["content"]
        text = "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict)
        )
        usage: dict[str, Any] = response.get("usage", {})
        metrics: dict[str, Any] = response.get("metrics", {})
        return FrontierResult(
            raw_output=text,
            stop_reason=str(response.get("stopReason", "unknown")),
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            total_tokens=int(usage.get("totalTokens", 0)),
            latency_ms=(
                int(metrics["latencyMs"])
                if metrics.get("latencyMs") is not None
                else None
            ),
        )

    def generate(self, prompts: list[str]) -> list[FrontierResult]:
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(executor.map(self._generate_one, prompts))
