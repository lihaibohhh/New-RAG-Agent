"""混合召回内部组件的最小接口。"""
from __future__ import annotations

from typing import Any, Protocol

from knowledge.contracts import RagDocument


class VectorCandidateRetrieverPort(Protocol):
    def retrieve(self, query: str) -> list[RagDocument]: ...

    def list_documents(self) -> list[RagDocument]: ...

    def warmup(self) -> None: ...


class DocumentCorpusPort(Protocol):
    def list_documents(self) -> list[RagDocument]: ...


class ChunkCorpusPort(DocumentCorpusPort, Protocol):
    def prepare(self) -> None: ...


class Bm25RepositoryPort(Protocol):
    def load(self) -> Any | None: ...

    def save(self, backend: Any, document_count: int) -> None: ...

    def clear(self) -> None: ...


__all__ = [
    "Bm25RepositoryPort",
    "ChunkCorpusPort",
    "DocumentCorpusPort",
    "VectorCandidateRetrieverPort",
]
