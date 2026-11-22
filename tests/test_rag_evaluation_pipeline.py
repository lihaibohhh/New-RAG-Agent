from __future__ import annotations

import pytest

from knowledge.contracts import KnowledgeDocument
from knowledge.rag.application import RetrievalApplicationService


def _document(chunk_id: str, content: str | None = None) -> KnowledgeDocument:
    return KnowledgeDocument(
        content=content or f"{chunk_id} 正文",
        metadata={
            "chunk_id": chunk_id,
            "source_file": "report.pdf",
            "page": 3,
            "doc_type": "text",
            "industry": "半导体",
        },
    )


class _Vector:
    def __init__(self, documents: list[KnowledgeDocument]) -> None:
        self._documents = documents

    def retrieve(self, _query: str) -> list[KnowledgeDocument]:
        return list(self._documents)

    def list_documents(self) -> list[KnowledgeDocument]:
        return list(self._documents)

    def warmup(self) -> None:
        return None


class _Bm25:
    def __init__(self, documents: list[KnowledgeDocument]) -> None:
        self._documents = documents

    def retrieve(self, _query: str) -> list[KnowledgeDocument]:
        return list(self._documents)

    def prepare(self) -> bool:
        return True

    def invalidate(self) -> None:
        return None

    async def close(self) -> None:
        return None


class _NullCache:
    async def get(self, _query: str):
        return None

    async def set(self, _query: str, _documents, *, top_score: float) -> None:
        return None


class _PassReranker:
    async def rerank(self, _query, documents, *, top_n):
        return documents[:top_n], 0.0

    async def warmup(self) -> None:
        return None


@pytest.mark.asyncio
async def test_hybrid_trace_preserves_each_candidate_stage() -> None:
    shared = _document("shared")
    service = RetrievalApplicationService(
        cache=_NullCache(),
        vector_retriever=_Vector([shared, _document("vector-only")]),
        bm25_retriever=_Bm25([_document("bm25-only"), shared]),
        reranker=_PassReranker(),
    )

    trace = await service._retrieve_with_trace(
        "问题", filters=None, mode="hybrid"
    )
    online_result = await service.search("问题", top_k=3)

    assert [item.metadata.chunk_id for item in trace.bm25_candidates] == [
        "bm25-only",
        "shared",
    ]
    assert [item.metadata.chunk_id for item in trace.vector_candidates] == [
        "shared",
        "vector-only",
    ]
    assert trace.fusion_candidates[0].metadata.chunk_id == "shared"
    assert [item.metadata.chunk_id for item in trace.filtered_candidates] == [
        item.chunk_id for item in online_result.chunks
    ]
    assert trace.degraded_sources == ()
    assert {"bm25", "vector", "fusion", "filter", "total"} <= trace.timings.keys()


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
    service = RetrievalApplicationService(
        cache=_NullCache(),
        bm25_retriever=_Bm25([documents[0]]),
        vector_retriever=_Vector([documents[1]]),
        reranker=reranker,
        max_content_chars=800,
        rerank_candidates=12,
        rerank_top_n=5,
        reranker_batch_size=2,
        device="cpu",
    )

    result = await service.evaluation_search("问题", top_k=1)

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
        await service.evaluation_search("问题", use_query_cache=True)
