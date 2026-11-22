"""Knowledge HTTP 请求、响应结构及领域对象转换。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from knowledge.contracts import (
    EvaluationCandidate,
    EvaluationRetrievalResult,
    IngestionReport,
    RagHealthStatus,
    RetrievedChunk,
    RetrievalResult,
    StoredChunk,
    WarmupStatus,
)


class _TransportModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SearchRequest(_TransportModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    filters: dict[str, Any] | None = None
    use_query_cache: bool = True
    retrieval_mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


class EvaluationSearchRequest(_TransportModel):
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


class TransportPayloadError(RuntimeError):
    """远端 Payload 不符合当前版本的传输契约。"""


_SchemaT = TypeVar("_SchemaT", bound=BaseModel)


def _validate(
    schema_type: type[_SchemaT],
    payload: Mapping[str, Any],
) -> _SchemaT:
    try:
        return schema_type.model_validate(payload)
    except ValidationError as exc:
        raise TransportPayloadError(
            f"Knowledge Service Payload 不符合 {schema_type.__name__}: {exc}"
        ) from exc


def _chunk_schema(chunk: RetrievedChunk) -> RetrievedChunkPayload:
    return RetrievedChunkPayload(
        content=chunk.content,
        source_file=chunk.source_file,
        source_page=chunk.source_page,
        chunk_id=chunk.chunk_id,
        score=chunk.score,
        doc_type=chunk.doc_type,
        industry=chunk.industry,
    )


def _chunk_domain(chunk: RetrievedChunkPayload) -> RetrievedChunk:
    return RetrievedChunk(
        content=chunk.content,
        source_file=chunk.source_file,
        source_page=chunk.source_page,
        chunk_id=chunk.chunk_id,
        score=chunk.score,
        doc_type=chunk.doc_type,
        industry=chunk.industry,
    )


def retrieval_result_to_payload(result: RetrievalResult) -> dict[str, Any]:
    schema = RetrievalResponse(
        query=result.query,
        chunks=[_chunk_schema(chunk) for chunk in result.chunks],
        stage=result.stage,
        cache_hit=result.cache_hit,
        candidates_count=result.candidates_count,
        reranked_count=result.reranked_count,
        top_score=result.top_score,
        timings=dict(result.timings),
    )
    return schema.model_dump(mode="json")


def retrieval_result_from_payload(
    payload: Mapping[str, Any],
    *,
    fallback_query: str = "",
) -> RetrievalResult:
    schema = _validate(RetrievalResponse, payload)
    return RetrievalResult(
        query=schema.query or fallback_query,
        chunks=tuple(_chunk_domain(chunk) for chunk in schema.chunks),
        stage=schema.stage,
        cache_hit=schema.cache_hit,
        candidates_count=schema.candidates_count,
        reranked_count=schema.reranked_count,
        top_score=schema.top_score,
        timings=dict(schema.timings),
    )


def _candidate_schema(
    candidate: EvaluationCandidate,
) -> EvaluationCandidatePayload:
    return EvaluationCandidatePayload(
        rank=candidate.rank,
        chunk_id=candidate.chunk_id,
        source_file=candidate.source_file,
        source_page=candidate.source_page,
        content_chars=candidate.content_chars,
        doc_type=candidate.doc_type,
        industry=candidate.industry,
    )


def _candidate_domain(
    candidate: EvaluationCandidatePayload,
) -> EvaluationCandidate:
    return EvaluationCandidate(
        rank=candidate.rank,
        chunk_id=candidate.chunk_id,
        source_file=candidate.source_file,
        source_page=candidate.source_page,
        content_chars=candidate.content_chars,
        doc_type=candidate.doc_type,
        industry=candidate.industry,
    )


def evaluation_result_to_payload(
    result: EvaluationRetrievalResult,
) -> dict[str, Any]:
    schema = EvaluationRetrievalResponse(
        query=result.query,
        retrieval_mode=result.retrieval_mode,
        chunks=[_chunk_schema(chunk) for chunk in result.chunks],
        trace=EvaluationTracePayload(
            stages={
                name: [_candidate_schema(candidate) for candidate in candidates]
                for name, candidates in result.stages.items()
            },
            timings=dict(result.timings),
            configuration=dict(result.configuration),
            degraded_sources=list(result.degraded_sources),
        ),
    )
    return schema.model_dump(mode="json")


def evaluation_result_from_payload(
    payload: Mapping[str, Any],
    *,
    fallback_query: str = "",
    fallback_retrieval_mode: str = "hybrid",
) -> EvaluationRetrievalResult:
    schema = _validate(EvaluationRetrievalResponse, payload)
    return EvaluationRetrievalResult(
        query=schema.query or fallback_query,
        retrieval_mode=schema.retrieval_mode or fallback_retrieval_mode,
        chunks=tuple(_chunk_domain(chunk) for chunk in schema.chunks),
        stages={
            name: tuple(_candidate_domain(candidate) for candidate in candidates)
            for name, candidates in schema.trace.stages.items()
        },
        timings=dict(schema.trace.timings),
        configuration=dict(schema.trace.configuration),
        degraded_sources=tuple(schema.trace.degraded_sources),
    )


def health_status_to_payload(status: RagHealthStatus) -> dict[str, Any]:
    return HealthResponse(
        ready=status.ready,
        state=status.state,
        details=dict(status.details),
    ).model_dump(mode="json")


def health_status_from_payload(payload: Mapping[str, Any]) -> RagHealthStatus:
    schema = _validate(HealthResponse, payload)
    return RagHealthStatus(
        ready=schema.ready,
        state=schema.state,
        details=dict(schema.details),
    )


def warmup_response_from_payload(
    payload: Mapping[str, Any],
) -> WarmupResponse:
    return _validate(WarmupResponse, payload)


def warmup_response_to_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = warmup_response_from_payload(payload)
    return schema.model_dump(mode="json", exclude_unset=True)


def warmup_status_from_payload(payload: Mapping[str, Any]) -> WarmupStatus:
    schema = warmup_response_from_payload(payload)
    if schema.ready is not True:
        reason = schema.warmup_status.error or schema.stage or "unknown"
        raise TransportPayloadError(f"Knowledge Service 未完成预热: {reason}")
    return WarmupStatus(
        ready=True,
        timings=dict(schema.warmup_status.timings),
    )


def ingestion_report_to_payload(report: IngestionReport) -> dict[str, Any]:
    return IngestionResponse(
        data_dir=report.data_dir,
        status=report.status,
        files_discovered=report.files_discovered,
        files_selected=report.files_selected,
        files_processed=report.files_processed,
        files_skipped=report.files_skipped,
        chunks_written=report.chunks_written,
        cache_invalidated=report.cache_invalidated,
        details=dict(report.details),
    ).model_dump(mode="json")


def ingestion_report_from_payload(
    payload: Mapping[str, Any],
    *,
    fallback_data_dir: str = "",
) -> IngestionReport:
    schema = _validate(IngestionResponse, payload)
    details = dict(schema.details)
    details.update(dict(schema.model_extra or {}))
    return IngestionReport(
        data_dir=schema.data_dir or fallback_data_dir,
        status=schema.status,
        files_discovered=schema.files_discovered,
        files_selected=schema.files_selected,
        files_processed=schema.files_processed,
        files_skipped=schema.files_skipped,
        chunks_written=schema.chunks_written,
        cache_invalidated=schema.cache_invalidated,
        details=details,
    )


@dataclass(frozen=True)
class DecodedChunkPage:
    items: tuple[StoredChunk, ...]
    offset: int
    limit: int
    has_more: bool


def stored_chunk_page_to_payload(
    items: tuple[StoredChunk, ...] | list[StoredChunk],
    *,
    offset: int,
    limit: int,
    has_more: bool,
) -> dict[str, Any]:
    schema = ChunkPageResponse(
        items=[
            StoredChunkPayload(
                chunk_id=item.chunk_id,
                content=item.content,
                metadata=item.metadata.to_dict(),
            )
            for item in items
        ],
        offset=offset,
        limit=limit,
        has_more=has_more,
    )
    return schema.model_dump(mode="json")


def stored_chunk_page_from_payload(
    payload: Mapping[str, Any],
) -> DecodedChunkPage:
    schema = _validate(ChunkPageResponse, payload)
    return DecodedChunkPage(
        items=tuple(
            StoredChunk(
                chunk_id=item.chunk_id,
                content=item.content,
                metadata=item.metadata,
            )
            for item in schema.items
        ),
        offset=schema.offset,
        limit=schema.limit,
        has_more=schema.has_more,
    )


__all__ = [
    "ChunkPageResponse",
    "DecodedChunkPage",
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
    "TransportPayloadError",
    "WarmupRequest",
    "WarmupResponse",
    "WarmupStatusPayload",
    "evaluation_result_from_payload",
    "evaluation_result_to_payload",
    "health_status_from_payload",
    "health_status_to_payload",
    "ingestion_report_from_payload",
    "ingestion_report_to_payload",
    "retrieval_result_from_payload",
    "retrieval_result_to_payload",
    "stored_chunk_page_from_payload",
    "stored_chunk_page_to_payload",
    "warmup_response_from_payload",
    "warmup_response_to_payload",
    "warmup_status_from_payload",
]
