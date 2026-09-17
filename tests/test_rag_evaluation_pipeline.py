from __future__ import annotations

import pytest

from react_agent.rag.contracts import CandidateRetrievalTrace, RagDocument
from react_agent.rag.evaluation import EvaluationRetrievalService
from react_agent.rag.infrastructure.retrieval.hybrid_retriever import (
    HybridRetrieverAdapter,
)


def _document(chunk_id: str, content: str | None = None) -> RagDocument:
    return RagDocument(
        content=content or f"{chunk_id} 正文",
        metadata={
            "chunk_id": chunk_id,
            "source_file": "report.pdf",
            "page": 3,
            "doc_type": "text",
            "industry": "半导体",
        },
    )


class _Repository:
    def load(self):
        return None

    def save(self, _backend, _document_count: int) -> None:
        return None

    def clear(self) -> None:
        return None


class _Vector:
    def __init__(self, documents: list[RagDocument]) -> None:
        self._documents = documents

    def retrieve(self, _query: str) -> list[RagDocument]:
        return list(self._documents)

    def list_documents(self) -> list[RagDocument]:
        return list(self._documents)

    def warmup(self) -> None:
        return None


class _Bm25:
    def __init__(self, documents: list[RagDocument]) -> None:
        self._documents = documents

    def retrieve(self, _query: str) -> list[RagDocument]:
        return list(self._documents)


@pytest.mark.asyncio
async def test_hybrid_trace_preserves_each_candidate_stage() -> None:
    shared = _document("shared")
    adapter = HybridRetrieverAdapter(
        vector_retriever=_Vector([shared, _document("vector-only")]),
        bm25_repository=_Repository(),
    )
    adapter._bm25 = _Bm25([_document("bm25-only"), shared])

    trace = await adapter.retrieve_with_trace("问题")
    online_result = await adapter.retrieve("问题")
    await adapter.close()

    assert [item.metadata.chunk_id for item in trace.bm25_candidates] == [
        "bm25-only",
        "shared",
    ]
    assert [item.metadata.chunk_id for item in trace.vector_candidates] == [
        "shared",
        "vector-only",
    ]
    assert trace.fusion_candidates[0].metadata.chunk_id == "shared"
    assert list(trace.filtered_candidates) == online_result
    assert trace.degraded_sources == ()
    assert {"bm25", "vector", "fusion", "filter", "total"} <= trace.timings.keys()


class _TraceRetriever:
    def __init__(self, documents: list[RagDocument]) -> None:
        self.documents = documents

    async def retrieve_with_trace(self, _query: str, **_kwargs):
        return CandidateRetrievalTrace(
            retrieval_mode="hybrid",
            bm25_candidates=(self.documents[0],),
            vector_candidates=(self.documents[1],),
            fusion_candidates=tuple(self.documents),
            filtered_candidates=tuple(self.documents),
            timings={"bm25": 0.01, "vector": 0.02, "fusion": 0.001},
        )


class _ReverseReranker:
    def __init__(self) -> None:
        self.top_n = 0

    async def rerank(self, _query, documents, *, top_n):
        self.top_n = top_n
        return list(reversed(documents))[:top_n], 0.9

    async def warmup(self) -> None:
        return None


@pytest.mark.asyncio
async def test_evaluation_pipeline_is_cache_free_and_returns_lightweight_trace() -> None:
    documents = [_document("bm25"), _document("vector")]
    reranker = _ReverseReranker()
    service = EvaluationRetrievalService(
        retriever=_TraceRetriever(documents),
        reranker=reranker,
        max_content_chars=800,
        rerank_candidates=12,
        production_rerank_top_n=5,
        reranker_batch_size=2,
        device="cpu",
    )

    result = await service.search("问题", top_k=1)

    assert result.chunks[0].chunk_id == "vector"
    assert result.stages["bm25"][0].chunk_id == "bm25"
    assert result.stages["reranker_input"][1].chunk_id == "vector"
    assert result.stages["reranked"][0].chunk_id == "vector"
    assert result.stages["final"][0].chunk_id == "vector"
    assert result.stages["bm25"][0].content_chars == len("bm25 正文")
    assert not hasattr(result.stages["bm25"][0], "content")
    assert result.configuration["query_cache_enabled"] is False
    assert result.configuration["device"] == "cpu"
    assert result.configuration["reranker_batch_size"] == 2
    assert reranker.top_n == 2

    with pytest.raises(ValueError, match="不允许使用语义查询缓存"):
        await service.search("问题", use_query_cache=True)
