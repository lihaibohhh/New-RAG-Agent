from __future__ import annotations

import pytest

from knowledge.client.remote import KnowledgeServiceClient
from knowledge.contracts import (
    EvaluationCandidate,
    EvaluationRetrievalResult,
    IngestionReport,
    KnowledgeValidationError,
    RagHealthStatus,
    RetrievedChunk,
    RetrievalResult,
    StoredChunk,
)
from knowledge.server import models as server_models
from knowledge.transport.codecs import (
    TransportPayloadError,
    evaluation_result_from_payload,
    evaluation_result_to_payload,
    health_status_from_payload,
    health_status_to_payload,
    ingestion_report_from_payload,
    ingestion_report_to_payload,
    retrieval_result_from_payload,
    retrieval_result_to_payload,
    stored_chunk_page_from_payload,
    stored_chunk_page_to_payload,
    warmup_response_to_payload,
    warmup_status_from_payload,
)
from knowledge.transport.schemas import SearchRequest


def test_retrieval_result_round_trip_preserves_all_fields() -> None:
    result = RetrievalResult(
        query="资本开支",
        chunks=(
            RetrievedChunk(
                content="证据正文",
                source_file="report.pdf",
                source_page=8,
                chunk_id="chunk-8",
                score=0.91,
                doc_type="text",
                industry="半导体",
            ),
        ),
        stage="reranked",
        cache_hit=True,
        candidates_count=20,
        reranked_count=5,
        top_score=0.91,
        timings={"total": 0.12},
    )

    payload = retrieval_result_to_payload(result)

    assert payload["chunks"][0] == {
        "content": "证据正文",
        "source_file": "report.pdf",
        "source_page": 8,
        "chunk_id": "chunk-8",
        "score": 0.91,
        "doc_type": "text",
        "industry": "半导体",
    }
    assert retrieval_result_from_payload(payload) == result


def test_evaluation_result_round_trip_preserves_trace() -> None:
    result = EvaluationRetrievalResult(
        query="先进制程",
        retrieval_mode="hybrid",
        chunks=(
            RetrievedChunk(
                content="正文",
                source_file="fab.pdf",
                source_page=None,
                chunk_id="fab-1",
            ),
        ),
        stages={
            "bm25": (
                EvaluationCandidate(
                    rank=1,
                    chunk_id="fab-1",
                    source_file="fab.pdf",
                    source_page=None,
                    content_chars=2,
                ),
            )
        },
        timings={"bm25": 0.01},
        configuration={"source_top_k": 10},
        degraded_sources=("vector_error",),
    )

    payload = evaluation_result_to_payload(result)

    assert payload["trace"]["stages"]["bm25"][0]["content_chars"] == 2
    assert evaluation_result_from_payload(payload) == result


def test_ingestion_report_round_trip_and_forward_compatible_extras() -> None:
    report = IngestionReport(
        data_dir="reports",
        status="completed",
        files_discovered=3,
        files_selected=2,
        files_processed=2,
        files_skipped=1,
        chunks_written=42,
        cache_invalidated=True,
        details={"duration": 1.25},
    )

    assert ingestion_report_from_payload(ingestion_report_to_payload(report)) == report

    decoded = ingestion_report_from_payload(
        {
            "data_dir": "reports",
            "status": "completed",
            "details": {"duration": 1.25},
            "future_metric": 7,
        }
    )
    assert decoded.details == {"duration": 1.25, "future_metric": 7}


def test_health_and_chunk_page_round_trip() -> None:
    health = RagHealthStatus(
        ready=True,
        state="knowledge_base_ready",
        details={"knowledge_base": {"chunk_count": 1}},
    )
    chunks = (
        StoredChunk(
            chunk_id="chunk-1",
            content="正文",
            metadata={"source_file": "report.pdf", "page": 3},
        ),
    )

    assert health_status_from_payload(health_status_to_payload(health)) == health
    page = stored_chunk_page_from_payload(
        stored_chunk_page_to_payload(
            chunks,
            offset=10,
            limit=1,
            has_more=True,
        )
    )
    assert page.items == chunks
    assert (page.offset, page.limit, page.has_more) == (10, 1, True)


def test_warmup_response_preserves_shape_and_builds_domain_status() -> None:
    payload = {
        "ready": True,
        "stage": "ready_after_wait",
        "waited_seconds": 0.2,
        "warmup_status": {
            "state": "done",
            "started_at": 1.0,
            "finished_at": 2.0,
            "timings": {"total": 1.0},
            "error": None,
        },
    }

    assert warmup_response_to_payload(payload) == payload
    status = warmup_status_from_payload(payload)
    assert status.ready is True
    assert status.timings == {"total": 1.0}


def test_invalid_remote_payload_has_stable_transport_error() -> None:
    with pytest.raises(TransportPayloadError, match="RetrievalResponse"):
        retrieval_result_from_payload(
            {"chunks": [{"source_page": "not-an-integer"}]}
        )


def test_server_request_models_reexport_transport_schema() -> None:
    assert server_models.SearchRequest is SearchRequest


@pytest.mark.asyncio
async def test_client_request_schema_errors_remain_knowledge_validation_errors() -> None:
    client = KnowledgeServiceClient(base_url="http://knowledge.test")
    try:
        with pytest.raises(KnowledgeValidationError, match="SearchRequest"):
            await client.search("", top_k=0)
    finally:
        await client.close()
