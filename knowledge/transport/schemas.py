"""不依赖 FastAPI、httpx 或模块内部实现的 HTTP 数据结构。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _TransportModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SearchRequest(_TransportModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    filters: dict[str, Any] | None = None
    use_query_cache: bool = True
    retrieval_mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


class EvaluationSearchRequest(_TransportModel):
    """专用评测检索请求；设计上不提供语义查询缓存开关。"""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    filters: dict[str, Any] | None = None
    retrieval_mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


class WarmupRequest(_TransportModel):
    wait_seconds: float = Field(default=20, ge=0, le=120)
    force: bool = False


class WarmupStatusPayload(_TransportModel):
    state: str = "not_started"
    started_at: float | None = None
    finished_at: float | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class WarmupResponse(_TransportModel):
    task_state: str | None = None
    ready: bool | None = None
    stage: str | None = None
    retry_after_seconds: float | None = None
    waited_seconds: float | None = None
    warmup_status: WarmupStatusPayload = Field(default_factory=WarmupStatusPayload)


class IngestionRequest(_TransportModel):
    """相对于服务端允许目录的路径，禁止客户端传宿主机绝对路径。"""

    relative_path: str = Field(default=".", min_length=1, max_length=1000)


class InvalidateRequest(_TransportModel):
    knowledge_base_id: str = Field(default="default", max_length=100)


class RetrievedChunkPayload(_TransportModel):
    content: str = ""
    source_file: str = ""
    source_page: int | None = None
    chunk_id: str = ""
    score: float | None = None
    doc_type: str | None = None
    industry: str | None = None


class RetrievalResponse(_TransportModel):
    query: str = ""
    chunks: list[RetrievedChunkPayload] = Field(default_factory=list)
    stage: str = "remote"
    cache_hit: bool = False
    candidates_count: int = 0
    reranked_count: int = 0
    top_score: float = 0.0
    timings: dict[str, float] = Field(default_factory=dict)


class EvaluationCandidatePayload(_TransportModel):
    rank: int = 0
    chunk_id: str = ""
    source_file: str = ""
    source_page: int | None = None
    content_chars: int = 0
    doc_type: str | None = None
    industry: str | None = None


class EvaluationTracePayload(_TransportModel):
    stages: dict[str, list[EvaluationCandidatePayload]] = Field(
        default_factory=dict
    )
    timings: dict[str, float] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)
    degraded_sources: list[str] = Field(default_factory=list)


class EvaluationRetrievalResponse(_TransportModel):
    query: str = ""
    retrieval_mode: str = ""
    chunks: list[RetrievedChunkPayload] = Field(default_factory=list)
    trace: EvaluationTracePayload = Field(default_factory=EvaluationTracePayload)


class HealthResponse(_TransportModel):
    ready: bool = False
    state: str = "unknown"
    details: dict[str, Any] = Field(default_factory=dict)


class StoredChunkPayload(_TransportModel):
    chunk_id: str = ""
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkPageResponse(_TransportModel):
    items: list[StoredChunkPayload] = Field(default_factory=list)
    offset: int = 0
    limit: int = 0
    has_more: bool = False


class IngestionResponse(_TransportModel):
    model_config = ConfigDict(extra="allow")

    data_dir: str = ""
    status: str = "unknown"
    files_discovered: int = 0
    files_selected: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    chunks_written: int = 0
    cache_invalidated: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ChunkPageResponse",
    "EvaluationCandidatePayload",
    "EvaluationRetrievalResponse",
    "EvaluationSearchRequest",
    "EvaluationTracePayload",
    "HealthResponse",
    "IngestionRequest",
    "IngestionResponse",
    "InvalidateRequest",
    "RetrievedChunkPayload",
    "RetrievalResponse",
    "SearchRequest",
    "StoredChunkPayload",
    "WarmupRequest",
    "WarmupResponse",
    "WarmupStatusPayload",
]
