"""BM25 retriever with per-chunk provenance metadata prefix.

Wraps InMemoryAdapter and rewrites each returned chunk's text to be:
    [session=<sid> speaker=<user_id> time=<occur_ts>] <original text>

Used for the c1_rag_provenance ablation (upstream Wave-1) — tests
whether explicit provenance hints help D1_conflict / D4_permission
retrieval, or hurt other dims.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import RetrievalAdapter
from .inmemory import InMemoryAdapter
from ..types import MessageEntry, SearchHit


class InMemProvenanceAdapter(RetrievalAdapter):
    """BM25 retrieval with per-chunk provenance metadata prefix."""

    def __init__(self) -> None:
        self._bm25 = InMemoryAdapter()

    def set_ego_session_map(self, ego_session_map: Dict[str, List[str]]) -> None:
        self._bm25.set_ego_session_map(ego_session_map)

    async def add(self, *, namespace: str, messages: List[MessageEntry]) -> dict:
        return await self._bm25.add(namespace=namespace, messages=messages)

    async def search(
        self,
        *,
        namespace: str,
        query: str,
        top_k: int,
        ego_id: Optional[str] = None,
    ) -> List[SearchHit]:
        hits = await self._bm25.search(
            namespace=namespace, query=query, top_k=top_k, ego_id=ego_id,
        )
        out = []
        for h in hits:
            speaker = f"u_{h.user_id}" if h.user_id is not None else "u_unknown"
            tagged_text = (
                f"[session={h.thread_id or '?'} speaker={speaker} time={h.occur_ts or '?'}] "
                f"{h.text}"
            )
            out.append(SearchHit(
                msg_id=h.msg_id,
                score=h.score,
                text=tagged_text,
                occur_ts=h.occur_ts,
                thread_id=h.thread_id,
                user_id=h.user_id,
            ))
        return out

    async def reset(self, *, namespace: str) -> None:
        await self._bm25.reset(namespace=namespace)

    async def close(self) -> None:
        await self._bm25.close()
