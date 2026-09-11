"""金融研报 RAG 模块的公共入口。"""
from __future__ import annotations

from react_agent.rag.contracts import (
    ChunkMetadata,
    IngestionReport,
    OcrPolicy,
    ParserHint,
    ParsedChunk,
    ParseRequest,
    ParseResult,
    RagDocument,
    RagHealthStatus,
    RetrievedChunk,
    RetrievalRequest,
    RetrievalResult,
    StoredChunk,
    SourceReference,
)


def build_vector_db(data_dir: str | None = None) -> IngestionReport:
    """兼容原同步建库入口，内部统一交给 IngestionService。"""
    from react_agent.core.config import settings
    from react_agent.rag.runtime import (
        RemoteRagRuntime,
        create_configured_rag_runtime,
    )

    runtime = create_configured_rag_runtime()
    target = (
        data_dir
        if data_dir is not None
        else "." if isinstance(runtime, RemoteRagRuntime)
        else settings.tools.vector_store.data_dir
    )
    return runtime.get_ingestion_service().ingest_sync(target)


__all__ = [
    "ChunkMetadata",
    "IngestionReport",
    "OcrPolicy",
    "ParserHint",
    "ParsedChunk",
    "ParseRequest",
    "ParseResult",
    "RagDocument",
    "RagHealthStatus",
    "RetrievedChunk",
    "RetrievalRequest",
    "RetrievalResult",
    "StoredChunk",
    "SourceReference",
    "build_vector_db",
]
