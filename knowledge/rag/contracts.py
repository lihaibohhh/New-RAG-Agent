"""仅在 RAG 模块内部流转的数据契约。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from knowledge.contracts import KnowledgeDocument, KnowledgeValidationError


@dataclass(frozen=True)
class RetrievalRequest:
    """一次在线检索用例的规范化请求。"""

    query: str
    top_k: int = 3
    filters: dict[str, Any] | None = None

    def normalized(self) -> "RetrievalRequest":
        query = (self.query or "").strip()
        if not query:
            raise KnowledgeValidationError("检索词不能为空")
        try:
            top_k = int(self.top_k)
        except (TypeError, ValueError) as exc:
            raise KnowledgeValidationError("top_k 必须是整数") from exc
        return RetrievalRequest(
            query=query,
            top_k=max(1, min(top_k, 10)),
            filters=dict(self.filters or {}) or None,
        )


@dataclass(frozen=True)
class CandidateRetrievalTrace:
    """BM25、向量与融合阶段的 RAG 内部候选快照。"""

    retrieval_mode: str
    bm25_candidates: tuple[KnowledgeDocument, ...] = ()
    vector_candidates: tuple[KnowledgeDocument, ...] = ()
    fusion_candidates: tuple[KnowledgeDocument, ...] = ()
    filtered_candidates: tuple[KnowledgeDocument, ...] = ()
    degraded_sources: tuple[str, ...] = ()
    timings: dict[str, float] = field(default_factory=dict)


__all__ = ["CandidateRetrievalTrace", "RetrievalRequest"]
