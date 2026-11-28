from __future__ import annotations

import pytest

from knowledge.contracts import KnowledgeDocument
from knowledge.rag.application import RetrievalApplicationService


class FakeCache:
    def __init__(self, cached: list[KnowledgeDocument] | None = None) -> None:
        self.cached = cached
        self.get_calls = 0
        self.set_calls = 0

    async def get(self, query: str) -> list[KnowledgeDocument] | None:
        self.get_calls += 1
        return self.cached

    async def set(
        self,
        query: str,
        documents: list[KnowledgeDocument],
        *,
        top_score: float,
    ) -> None:
        self.set_calls += 1

    async def clear(self) -> None:
        return None


class FakeVectorRetriever:
    def __init__(self, document: KnowledgeDocument) -> None:
        self.document = document
        self.calls = 0

    def retrieve(self, query: str) -> list[KnowledgeDocument]:
        self.calls += 1
        return [self.document]

    def warmup(self) -> None:
        return None


class FakeBm25Retriever(FakeVectorRetriever):
    def __init__(self, document: KnowledgeDocument) -> None:
        super().__init__(document)
        self.prepare_calls = 0

    def prepare(self) -> bool:
        self.prepare_calls += 1
        return True

    def invalidate(self) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeReranker:
    async def rerank(
        self,
        query: str,
        documents: list[KnowledgeDocument],
        *,
        top_n: int,
    ) -> tuple[list[KnowledgeDocument], float]:
        return documents[:top_n], 0.9

    async def warmup(self) -> None:
        return None


@pytest.mark.asyncio
async def test_evaluation_can_bypass_query_cache() -> None:
    cached = KnowledgeDocument(
        content="旧缓存",
        metadata={"source_file": "old.pdf", "page": 1, "chunk_id": "old"},
    )
    fresh = KnowledgeDocument(
        content="新检索结果",
        metadata={"source_file": "new.pdf", "page": 2, "chunk_id": "new"},
    )
    cache = FakeCache([cached])
    vector = FakeVectorRetriever(fresh)
    bm25 = FakeBm25Retriever(fresh)
    service = RetrievalApplicationService(
        cache=cache,
        vector_retriever=vector,
        bm25_retriever=bm25,
        reranker=FakeReranker(),
    )

    result = await service.search("问题", top_k=3, use_query_cache=False)

    assert result.cache_hit is False
    assert result.chunks[0].chunk_id == "new"
    assert cache.get_calls == 0
    assert cache.set_calls == 0
    assert vector.calls == 1
    assert bm25.calls == 1


@pytest.mark.asyncio
async def test_online_default_still_uses_query_cache() -> None:
    cached = KnowledgeDocument(
        content="缓存结果",
        metadata={"source_file": "cached.pdf", "page": 1, "chunk_id": "cached"},
    )
    cache = FakeCache([cached])
    vector = FakeVectorRetriever(cached)
    bm25 = FakeBm25Retriever(cached)
    service = RetrievalApplicationService(
        cache=cache,
        vector_retriever=vector,
        bm25_retriever=bm25,
        reranker=FakeReranker(),
    )

    result = await service.search("问题", top_k=3)

    assert result.cache_hit is True
    assert cache.get_calls == 1
    assert cache.set_calls == 0
    assert vector.calls == 0
    assert bm25.calls == 0
    assert bm25.prepare_calls == 0


@pytest.mark.asyncio
async def test_single_source_evaluation_never_reuses_hybrid_query_cache() -> None:
    cached = KnowledgeDocument(
        content="混合缓存",
        metadata={"source_file": "cached.pdf", "page": 1, "chunk_id": "cached"},
    )
    bm25 = KnowledgeDocument(
        content="BM25 结果",
        metadata={"source_file": "bm25.pdf", "page": 2, "chunk_id": "bm25"},
    )
    cache = FakeCache([cached])
    vector = FakeVectorRetriever(cached)
    bm25_retriever = FakeBm25Retriever(bm25)
    service = RetrievalApplicationService(
        cache=cache,
        vector_retriever=vector,
        bm25_retriever=bm25_retriever,
        reranker=FakeReranker(),
    )

    result = await service.search(
        "问题",
        top_k=3,
        use_query_cache=True,
        retrieval_mode="bm25",
    )

    assert result.chunks[0].chunk_id == "bm25"
    assert result.stage == "bm25_retrieve+rerank"
    assert cache.get_calls == 0
    assert cache.set_calls == 0
    assert bm25_retriever.prepare_calls == 1
    assert bm25_retriever.calls == 1
    assert vector.calls == 0
