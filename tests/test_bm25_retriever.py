from __future__ import annotations

import fakeredis

from knowledge.contracts import RagDocument
from knowledge.infrastructure.retrieval.bm25_repository import (
    RedisBm25Repository,
)
from knowledge.infrastructure.retrieval.bm25_retriever import (
    Bm25CandidateRetriever,
)
from knowledge.infrastructure.retrieval.bm25_tokenizer import (
    bm25_tokenizer_fingerprint,
    normalize_bm25_text,
    tokenize_bm25,
)


def _document(chunk_id: str, content: str) -> RagDocument:
    return RagDocument(
        content=content,
        metadata={
            "chunk_id": chunk_id,
            "source_file": f"{chunk_id}.pdf",
            "page": 1,
        },
    )


def test_chinese_search_tokenizer_preserves_finance_and_model_terms() -> None:
    tokens = tokenize_bm25(
        "ＴＳＭＣ预计2026H1资本开支为520-560亿美元，HBM4带宽3.6TB/s。"
    )

    assert normalize_bm25_text("ＴＳＭＣ HBM4") == "tsmc hbm4"
    assert "资本开支" in tokens
    assert "2026h1" in tokens
    assert "hbm4" in tokens
    assert "3.6tb/s" in tokens
    assert "520-560" in tokens


def test_bm25_chinese_query_returns_relevant_document() -> None:
    retriever = Bm25CandidateRetriever.build(
        [
            _document("capex", "台积电2026年资本开支指引为520至560亿美元。"),
            _document("education", "高等教育毕业生规模继续增长。"),
            _document("property", "房地产市场商品房成交面积下降。"),
        ],
        top_k=3,
    )

    result = retriever.retrieve("台积电的2026年资本开支是多少？")

    assert result
    assert result[0].metadata.chunk_id == "capex"
    assert all(item.metadata.chunk_id != "education" for item in result)


def test_bm25_all_zero_scores_return_no_arbitrary_documents() -> None:
    retriever = Bm25CandidateRetriever.build(
        [
            _document("a", "资本开支和营业收入。"),
            _document("b", "房地产成交面积。"),
            _document("c", "高等教育招生人数。"),
        ],
        top_k=3,
    )

    assert retriever.retrieve("xyznonexistenttoken") == []


def test_bm25_redis_cache_rejects_legacy_tokenizer_version() -> None:
    redis = fakeredis.FakeRedis()
    repository = RedisBm25Repository(
        hmac_secret="test-secret",
        ttl=60,
        redis_provider=lambda: redis,
    )
    backend = {"index": "test"}
    repository.save(backend, 1)

    assert repository.load() == backend
    assert redis.get("rag:bm25_index_version").decode() == (
        bm25_tokenizer_fingerprint()
    )

    redis.set("rag:bm25_index_version", "legacy-tokenizer")

    assert repository.load() is None


def test_bm25_backend_with_custom_tokenizer_survives_redis_round_trip() -> None:
    redis = fakeredis.FakeRedis()
    repository = RedisBm25Repository(
        hmac_secret="test-secret",
        ttl=60,
        redis_provider=lambda: redis,
    )
    original = Bm25CandidateRetriever.build(
        [
            _document("capex", "台积电资本开支指引上调。"),
            _document("education", "高等教育毕业生规模增长。"),
            _document("property", "房地产成交面积下降。"),
        ],
        top_k=3,
    )
    repository.save(original.backend, 3)

    restored = Bm25CandidateRetriever(repository.load(), top_k=3)

    assert restored.retrieve("资本开支")[0].metadata.chunk_id == "capex"


def test_bm25_cache_clear_removes_tokenizer_version() -> None:
    redis = fakeredis.FakeRedis()
    repository = RedisBm25Repository(
        hmac_secret="test-secret",
        ttl=60,
        redis_provider=lambda: redis,
    )
    repository.save({"index": "test"}, 1)

    repository.clear()

    assert redis.get("rag:bm25_index") is None
    assert redis.get("rag:bm25_index_version") is None
