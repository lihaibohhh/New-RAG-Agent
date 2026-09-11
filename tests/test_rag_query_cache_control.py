from __future__ import annotations

import pytest

from react_agent.rag.contracts import RagDocument
from react_agent.rag.query import RetrievalService


class FakeCache:
    def __init__(self, cached: list[RagDocument] | None = None) -> None:
        self.cached = cached
        self.get_calls = 0
        self.set_calls = 0

    async def get(self, query: str) -> list[RagDocument] | None:
        self.get_calls += 1
        return self.cached

    async def set(
        self,
        query: str,
        documents: list[RagDocument],
        *,
        top_score: float,
    ) -> None:
        self.set_calls += 1

    async def clear(self) -> None:
        return None


class FakeRetriever:
    def __init__(self, document: RagDocument) -> None:
        self.document = document
        self.calls = 0

    async def retrieve(self, query: str, *, filters=None) -> list[RagDocument]:
        self.calls += 1
        return [self.document]

    async def warmup(self) -> None:
        return None

    async def invalidate(self) -> None:
        return None


class ModeAwareRetriever(FakeRetriever):
    def __init__(self, document: RagDocument) -> None:
        super().__init__(document)
        self.modes: list[str] = []

    async def retrieve(
        self,
        query: str,
        *,
        filters=None,
        mode: str = "hybrid",
    ) -> list[RagDocument]:
        self.calls += 1
        self.modes.append(mode)
        return [self.document]


class FakeReranker:
    async def rerank(
        self,
        query: str,
        documents: list[RagDocument],
        *,
        top_n: int,
    ) -> tuple[list[RagDocument], float]:
        return documents[:top_n], 0.9

    async def warmup(self) -> None:
        return None


@pytest.mark.asyncio
async def test_evaluation_can_bypass_query_cache() -> None:
    cached = RagDocument(
        content="旧缓存",
        metadata={"source_file": "old.pdf", "page": 1, "chunk_id": "old"},
    )
    fresh = RagDocument(
        content="新检索结果",
        metadata={"source_file": "new.pdf", "page": 2, "chunk_id": "new"},
    )
    cache = FakeCache([cached])
    retriever = FakeRetriever(fresh)
    service = RetrievalService(
        cache=cache,
        retriever=retriever,
        reranker=FakeReranker(),
    )

    result = await service.search("问题", top_k=3, use_query_cache=False)

    assert result.cache_hit is False
    assert result.chunks[0].chunk_id == "new"
    assert cache.get_calls == 0
    assert cache.set_calls == 0
    assert retriever.calls == 1


@pytest.mark.asyncio
async def test_online_default_still_uses_query_cache() -> None:
    cached = RagDocument(
        content="缓存结果",
        metadata={"source_file": "cached.pdf", "page": 1, "chunk_id": "cached"},
    )
    cache = FakeCache([cached])
    retriever = FakeRetriever(cached)
    service = RetrievalService(
        cache=cache,
        retriever=retriever,
        reranker=FakeReranker(),
    )

    result = await service.search("问题", top_k=3)

    assert result.cache_hit is True
    assert cache.get_calls == 1
    assert retriever.calls == 0


@pytest.mark.asyncio
async def test_single_source_evaluation_never_reuses_hybrid_query_cache() -> None:
    cached = RagDocument(
        content="混合缓存",
        metadata={"source_file": "cached.pdf", "page": 1, "chunk_id": "cached"},
    )
    bm25 = RagDocument(
        content="BM25 结果",
        metadata={"source_file": "bm25.pdf", "page": 2, "chunk_id": "bm25"},
    )
    cache = FakeCache([cached])
    retriever = ModeAwareRetriever(bm25)
    service = RetrievalService(
        cache=cache,
        retriever=retriever,
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
    assert retriever.modes == ["bm25"]
