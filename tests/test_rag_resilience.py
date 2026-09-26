from __future__ import annotations

import pytest

from knowledge.contracts import KnowledgeDocument
from knowledge.rag.infrastructure.retrieval.bm25_index import (
    LazyBm25CandidateRetrieverAdapter,
)
from knowledge.rag.infrastructure.retrieval.bm25_repository import (
    RedisBm25Repository,
)
from knowledge.rag.retrieval import HybridRetrievalService


class FakeRepository:
    def __init__(self) -> None:
        self.saved = 0
        self.cleared = 0

    def load(self):
        return None

    def save(self, _backend, _document_count: int) -> None:
        self.saved += 1

    def clear(self) -> None:
        self.cleared += 1


class FailingVector:
    def retrieve(self, _query: str):
        raise RuntimeError("hnsw unavailable")

    def list_documents(self):
        return []

    def warmup(self) -> None:
        raise RuntimeError("hnsw unavailable")


class FakeBm25:
    def __init__(self, document: KnowledgeDocument) -> None:
        self._document = document

    def retrieve(self, _query: str):
        return [self._document]

    def prepare(self) -> bool:
        return True

    def invalidate(self) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeCorpus:
    def prepare(self) -> None:
        return None

    def list_documents(self):
        return []


@pytest.mark.asyncio
async def test_hybrid_retrieval_degrades_to_bm25_when_hnsw_fails() -> None:
    document = KnowledgeDocument(
        content="BM25 仍可返回的证据",
        metadata={"source_file": "report.pdf", "chunk_id": "chunk-1"},
    )
    adapter = HybridRetrievalService(
        vector_retriever=FailingVector(),
        bm25_retriever=FakeBm25(document),
    )

    result = await adapter.retrieve("测试", mode="hybrid")
    await adapter.close()

    assert result == [document]


@pytest.mark.asyncio
async def test_stale_bm25_save_is_skipped_after_invalidation_generation() -> None:
    repository = FakeRepository()
    adapter = LazyBm25CandidateRetrieverAdapter(
        repository=repository,
        corpus_reader=FakeCorpus(),
    )
    adapter._generation = 2

    adapter._save_if_current(object(), 10, generation=1)
    await adapter.close()

    assert repository.saved == 0


def test_bm25_clear_failure_is_not_reported_as_success() -> None:
    def failing_redis():
        raise ConnectionError("redis unavailable")

    repository = RedisBm25Repository(
        hmac_secret="test-secret",
        ttl=60,
        redis_provider=failing_redis,
    )

    with pytest.raises(RuntimeError, match="缓存清理失败"):
        repository.clear()
