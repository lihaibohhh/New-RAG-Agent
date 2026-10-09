"""Knowledge 包对应用和协议适配器暴露的两个业务服务契约。"""

from __future__ import annotations

from typing import Any, Protocol

from knowledge.contracts import (
    EvaluationRetrievalResult,
    IngestionReport,
    RagHealthStatus,
    RetrievalResult,
    StoredChunk,
)


class RagService(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        use_query_cache: bool = True,
        retrieval_mode: str = "hybrid",
    ) -> RetrievalResult: ...

    async def evaluation_search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        retrieval_mode: str = "hybrid",
        use_query_cache: bool = False,
    ) -> EvaluationRetrievalResult: ...

    async def start_warmup(self, force: bool = False) -> dict[str, Any]: ...

    async def get_warmup_status(self) -> dict[str, Any]: ...

    async def ensure_ready(
        self,
        wait_seconds: int | float = 20,
    ) -> dict[str, Any]: ...

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

    async def close(self) -> None: ...


class IngestionService(Protocol):
    async def ingest(self, path: str, /) -> IngestionReport: ...

    def ingest_sync(self, path: str, /) -> IngestionReport: ...

    async def close(self) -> None: ...


__all__ = ["IngestionService", "RagService"]
