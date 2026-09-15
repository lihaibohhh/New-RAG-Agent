"""RAG Runtime 对应用与协议适配器公开的能力契约。"""
from __future__ import annotations

from typing import Any, Protocol

from react_agent.rag.contracts import (
    EvaluationRetrievalResult,
    IngestionReport,
    RagHealthStatus,
    RetrievalResult,
    StoredChunk,
    WarmupStatus,
)


class RetrievalServicePort(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        use_query_cache: bool = True,
        retrieval_mode: str = "hybrid",
    ) -> RetrievalResult: ...

    async def warmup(self) -> WarmupStatus: ...


class EvaluationRetrievalServicePort(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        retrieval_mode: str = "hybrid",
        use_query_cache: bool = False,
    ) -> EvaluationRetrievalResult: ...


class RagOperationsPort(Protocol):
    def start(self, force: bool = False) -> dict[str, Any]: ...

    def get_status(self) -> dict[str, Any]: ...

    async def ensure_ready(
        self,
        wait_seconds: int | float = 20,
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class RagAdminServicePort(Protocol):
    async def health(self) -> RagHealthStatus: ...

    async def invalidate(self, knowledge_base_id: str = "default") -> None: ...

    async def read_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredChunk, ...]: ...

    def read_chunks_sync(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredChunk, ...]: ...


class IngestionServicePort(Protocol):
    async def ingest(self, path: str, /) -> IngestionReport: ...

    def ingest_sync(self, path: str, /) -> IngestionReport: ...


class AgentRagRuntimePort(Protocol):
    """Application Runtime 使用的最小 RAG 能力视图。"""

    operations: RagOperationsPort

    def get_retrieval_service(self) -> RetrievalServicePort: ...

    async def close(self) -> None: ...


class RagRuntimePort(AgentRagRuntimePort, Protocol):
    """管理、评测和 MCP 入口使用的完整 RAG Runtime 契约。"""

    def get_evaluation_retrieval_service(
        self,
    ) -> EvaluationRetrievalServicePort: ...

    def get_admin_service(self) -> RagAdminServicePort: ...

    def get_ingestion_service(self) -> IngestionServicePort: ...


__all__ = [
    "AgentRagRuntimePort",
    "EvaluationRetrievalServicePort",
    "IngestionServicePort",
    "RagAdminServicePort",
    "RagOperationsPort",
    "RagRuntimePort",
    "RetrievalServicePort",
]
