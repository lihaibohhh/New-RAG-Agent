"""RAG 模块拥有、由内部服务和适配器实现的端口。"""

from __future__ import annotations

from typing import Any, Protocol

from knowledge.contracts import KnowledgeDocument, StoredChunk, WarmupStatus
from knowledge.rag.contracts import CandidateRetrievalTrace


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


class HybridRetrieverPort(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ) -> list[KnowledgeDocument]: ...

    async def warmup(self) -> None: ...

    async def invalidate(self) -> None: ...


class EvaluationHybridRetrieverPort(Protocol):
    async def retrieve_with_trace(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ) -> CandidateRetrievalTrace: ...


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


class KnowledgeBaseInspectorPort(Protocol):
    async def inspect(self) -> dict[str, Any]: ...


class Bm25RepositoryPort(Protocol):
    def load(self) -> Any | None: ...

    def save(self, backend: Any, document_count: int) -> None: ...

    def clear(self) -> None: ...


class RagCacheInvalidatorPort(Protocol):
    async def invalidate(self) -> None: ...


class Bm25CachePort(Protocol):
    def clear(self) -> None: ...


__all__ = [
    "Bm25CachePort",
    "Bm25CandidateRetrieverPort",
    "Bm25RepositoryPort",
    "ChunkCorpusPort",
    "ChunkReaderPort",
    "ChunkStorePort",
    "DocumentCorpusPort",
    "EvaluationHybridRetrieverPort",
    "HybridRetrieverPort",
    "KnowledgeBaseInspectorPort",
    "RagCacheInvalidatorPort",
    "RerankerPort",
    "SemanticCachePort",
    "VectorCandidateRetrieverPort",
    "WarmupServicePort",
]
