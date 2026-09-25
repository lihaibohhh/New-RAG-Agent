"""知识库应用层依赖的端口定义。"""
from __future__ import annotations

from typing import Any, Protocol

from knowledge.contracts import (
    CandidateRetrievalTrace,
    ParseRequest,
    ParseResult,
    RagDocument,
    StoredChunk,
)


class DocumentParserPort(Protocol):
    """任意单一文档解析实现遵循的最小端口。"""

    async def parse(self, request: ParseRequest) -> ParseResult: ...


class SemanticCachePort(Protocol):
    async def get(self, query: str) -> list[RagDocument] | None: ...

    async def set(
        self,
        query: str,
        documents: list[RagDocument],
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
    ) -> list[RagDocument]: ...

    async def warmup(self) -> None: ...

    async def invalidate(self) -> None: ...


class EvaluationHybridRetrieverPort(Protocol):
    """只在评测用例中暴露候选阶段，不扩展线上检索端口。"""

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
        documents: list[RagDocument],
        *,
        top_n: int,
    ) -> tuple[list[RagDocument], float]: ...

    async def warmup(self) -> None: ...


class VectorIndexWriterPort(Protocol):
    """向量索引的同步批量写入端口。

    建库应用服务负责扫描、解析、批次和失败策略；实现只负责准备底层
    向量存储并写入一个已经完成解析的批次。
    """

    def initialize(self) -> None: ...

    def write(
        self,
        documents: list[RagDocument],
        *,
        batch_label: str,
        global_offset: int,
    ) -> None: ...


class IngestionManifestPort(Protocol):
    """增量建库文件指纹清单端口。"""

    def load(self) -> dict[str, str]: ...

    def save(self, record: dict[str, str]) -> None: ...


class PdfPageCounterPort(Protocol):
    """PDF 页数读取端口，用于建库前的超长文档过滤。"""

    def count(self, file_path: str) -> int: ...


class IngestionPreflightPort(Protocol):
    """建库外部依赖的同步预检端口。"""

    def ensure_ready(self) -> None: ...


class RagCacheInvalidatorPort(Protocol):
    async def invalidate(self) -> None: ...


class KnowledgeIndexChangedPort(Protocol):
    """接收建库提交事件；建库用例不关心下游如何刷新缓存。"""

    async def notify_index_changed(self) -> None: ...


class CacheInvalidationCoordinatorPort(
    RagCacheInvalidatorPort,
    KnowledgeIndexChangedPort,
    Protocol,
):
    """组合根内部适配器契约，同时服务管理命令与索引变更事件。"""


class Bm25CachePort(Protocol):
    """BM25 持久化缓存的最小失效端口。"""

    def clear(self) -> None: ...


class KnowledgeBasePort(Protocol):
    """知识库轻量诊断和离线语料读取端口。"""

    async def inspect(self) -> dict[str, Any]: ...

    async def list_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[StoredChunk]: ...


__all__ = [
    "Bm25CachePort",
    "CacheInvalidationCoordinatorPort",
    "DocumentParserPort",
    "EvaluationHybridRetrieverPort",
    "HybridRetrieverPort",
    "IngestionManifestPort",
    "IngestionPreflightPort",
    "KnowledgeBasePort",
    "KnowledgeIndexChangedPort",
    "PdfPageCounterPort",
    "RagCacheInvalidatorPort",
    "RerankerPort",
    "SemanticCachePort",
    "VectorIndexWriterPort",
]
