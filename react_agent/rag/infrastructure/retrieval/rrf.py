"""纯函数形式的 Reciprocal Rank Fusion 策略。"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

from react_agent.rag.contracts import RagDocument


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[RagDocument]],
    *,
    rank_constant: int = 60,
    limit: int = 20,
) -> list[RagDocument]:
    scores: dict[str, float] = {}
    documents: dict[str, RagDocument] = {}
    for ranking in rankings:
        for rank, document in enumerate(ranking):
            key = _document_key(document)
            scores[key] = scores.get(key, 0.0) + 1 / (rank_constant + rank + 1)
            documents[key] = document
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [documents[key] for key in ordered[:limit]]


def _document_key(document: RagDocument) -> str:
    chunk_id = document.metadata.chunk_id or document.document_id
    if chunk_id not in (None, ""):
        return str(chunk_id)
    return hashlib.sha256(document.content.encode("utf-8")).hexdigest()


__all__ = ["reciprocal_rank_fusion"]
