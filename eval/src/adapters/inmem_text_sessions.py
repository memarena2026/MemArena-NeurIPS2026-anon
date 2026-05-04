"""InMemory (BM25) adapter with TEXT_SESSIONS prompt style.

Used for §27 P3: prompt-matched RAG × D5 control experiment.
Identical to InMemoryAdapter except prompt_style returns TEXT_SESSIONS
instead of JSON_CONTEXT, matching Oracle/Vanilla prompt formatting.
"""
from .inmemory import InMemoryAdapter
from .base import PromptStyle


class InMemTextSessionsAdapter(InMemoryAdapter):
    """BM25 retrieval with TEXT_SESSIONS prompt format (same as Oracle/Vanilla)."""

    @property
    def prompt_style(self) -> PromptStyle:
        return PromptStyle.TEXT_SESSIONS
