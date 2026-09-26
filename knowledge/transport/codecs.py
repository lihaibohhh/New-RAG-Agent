"""Knowledge 领域对象与 HTTP JSON Payload 的唯一转换实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel, ValidationError

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
from knowledge.transport.schemas import (
    ChunkPageResponse,
    EvaluationCandidatePayload,
    EvaluationRetrievalResponse,
    EvaluationTracePayload,
    HealthResponse,
    IngestionResponse,
    RetrievedChunkPayload,
    RetrievalResponse,
    StoredChunkPayload,
    WarmupResponse,
)


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
    "DecodedChunkPage",
    "TransportPayloadError",
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
