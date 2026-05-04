"""Shared text cleanup helpers for LLM outputs."""

from __future__ import annotations

import re


_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?>",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_RE = re.compile(r"^\s*```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```\s*$", re.DOTALL)
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:rephrased\s+question|paraphrased\s+question|rewritten\s+question|question)\s*:\s*",
    re.IGNORECASE,
)


def clean_llm_text(text: object) -> str:
    """Remove reasoning wrappers and common response scaffolding from LLM text.

    Qwen3-style reasoning models often return ``<think>...</think>`` before the
    actual answer. This helper is intentionally plain-text oriented: it keeps
    the final user-visible content and removes only wrappers/prefixes that are
    not part of the benchmark question or model answer.
    """
    cleaned = str(text or "").strip()
    cleaned = _THINK_BLOCK_RE.sub("", cleaned).strip()

    match = _FENCE_RE.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
        cleaned = _THINK_BLOCK_RE.sub("", cleaned).strip()

    cleaned = _LEADING_LABEL_RE.sub("", cleaned).strip()

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()

    return cleaned
