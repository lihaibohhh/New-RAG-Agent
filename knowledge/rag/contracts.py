"""仅在 RAG 模块内部流转的数据契约。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from knowledge.contracts import (
    KnowledgeDocument,
    KnowledgeValidationError,
    StoredChunk,
    WarmupStatus,
)


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


class SemanticCachePort(Protocol):
    async def get(self, query: str) -> list[KnowledgeDocument] | None: ...

    async def set(
        self,
        query: str,
        documents: list[KnowledgeDocument],
        *,
        top_score: float,
    ) -> None: ...

    async def clear(self) -> None: ...


class RerankerPort(Protocol):
    async def rerank(
        self,
        query: str,
        documents: list[KnowledgeDocument],
        *,
        top_n: int,
    ) -> tuple[list[KnowledgeDocument], float]: ...

    async def warmup(self) -> None: ...


class WarmupServicePort(Protocol):
    async def warmup(self) -> WarmupStatus: ...


class VectorCandidateRetrieverPort(Protocol):
    def retrieve(self, query: str) -> list[KnowledgeDocument]: ...

    def warmup(self) -> None: ...


class Bm25CandidateRetrieverPort(Protocol):
    def prepare(self) -> bool: ...

    def retrieve(self, query: str) -> list[KnowledgeDocument]: ...

    def invalidate(self) -> None: ...

    async def close(self) -> None: ...


class DocumentCorpusPort(Protocol):
    def list_documents(self) -> list[KnowledgeDocument]: ...


class ChunkCorpusPort(DocumentCorpusPort, Protocol):
    def prepare(self) -> None: ...


class ChunkReaderPort(Protocol):
    async def list_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[StoredChunk]: ...


class ChunkStorePort(DocumentCorpusPort, ChunkReaderPort, Protocol):
    def snapshot_complete(self) -> bool: ...

    def replace_all(self, documents: list[KnowledgeDocument]) -> None: ...


class Bm25RepositoryPort(Protocol):
    def load(self) -> Any | None: ...

    def save(self, backend: Any, document_count: int) -> None: ...

    def clear(self) -> None: ...


__all__ = [
    "Bm25CandidateRetrieverPort",
    "Bm25RepositoryPort",
    "CandidateRetrievalTrace",
    "ChunkCorpusPort",
    "ChunkReaderPort",
    "ChunkStorePort",
    "DocumentCorpusPort",
    "RerankerPort",
    "RetrievalRequest",
    "SemanticCachePort",
    "VectorCandidateRetrieverPort",
    "WarmupServicePort",
]
