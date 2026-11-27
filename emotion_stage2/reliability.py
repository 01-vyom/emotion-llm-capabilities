"""Shared reliability-signal helpers."""

from __future__ import annotations

import re


_REFUSAL_PATTERNS = (
    r"\bi (?:am|['’]m) unable to\b",
    r"\bi (?:cannot|can't|won't) (?:help|assist|comply|provide|complete)\b",
    r"\bi must refuse\b",
    r"\bi (?:cannot|can't) fulfill\b",
    r"\bas an ai\b.*\b(?:cannot|can't|unable)\b",
    r"\bnot able to (?:help|assist|comply|provide|complete)\b",
)
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), flags=re.IGNORECASE | re.DOTALL)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text))
