"""RAG 模块拥有的本地对象图与生命周期。"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, Literal

from knowledge.rag.admin import RagAdminService
from knowledge.rag.infrastructure.cache.semantic_cache import (
    RedisSemanticCacheAdapter,
)
from knowledge.rag.infrastructure.invalidation import RagCacheInvalidator
from knowledge.rag.infrastructure.redis import RedisClientManager
from knowledge.rag.infrastructure.retrieval.bm25_index import (
    LazyBm25CandidateRetrieverAdapter,
)
from knowledge.rag.infrastructure.retrieval.bm25_repository import (
    RedisBm25Repository,
)
from knowledge.rag.infrastructure.retrieval.reranker import (
    RerankerProviderAdapter,
)
from knowledge.rag.infrastructure.storage import ChromaKnowledgeBaseAdapter
from knowledge.rag.config import RagRuntimeConfig
from knowledge.rag.operations import RagWarmupManager
from knowledge.rag.ports import ChunkStorePort, HybridRetrieverPort
from knowledge.rag.query import RetrievalService
from knowledge.rag.retrieval import HybridRetrievalService

logger = logging.getLogger(__name__)


class LocalRagRuntime:
    """只组装并暴露在线检索、评测、管理和预热能力。"""

    def __init__(
        self,
        config: RagRuntimeConfig,
        *,
        chroma_dir: str,
        device_provider: Callable[[], Literal["cpu", "cuda"]],
        embedding_provider: Callable[[], Any],
        chunk_store_provider: Callable[[], ChunkStorePort],
        knowledge_write_lock: threading.RLock,
    ) -> None:
        self._config = config
        self._chroma_dir = chroma_dir
        self._device_provider = device_provider
        self._embedding_provider = embedding_provider
        self._chunk_store_provider = chunk_store_provider
        self._knowledge_write_lock = knowledge_write_lock
        self._retrieval_service: RetrievalService | None = None
        self._evaluation_retrieval_service = None
        self._admin_service: RagAdminService | None = None
        self._cache_invalidator: RagCacheInvalidator | None = None
        self._hybrid_retriever: HybridRetrieverPort | None = None
        self._vector_retriever = None
        self._chunk_corpus = None
        self._reranker = None
        self._lock = threading.RLock()
        self._redis_resources = RedisClientManager(
            url=config.redis.url,
            max_connections=config.redis.max_connections,
        )
        self.operations = RagWarmupManager(self.get_retrieval_service)

    def get_retrieval_service(self) -> RetrievalService:
        if self._retrieval_service is not None:
            return self._retrieval_service

        with self._lock:
            if self._retrieval_service is None:
                profile = self._config.tuning.profile(self._device_provider())
                logger.info(
                    "[RAG] runtime profile device=%s rerank_candidates=%s "
                    "rerank_top_n=%s reranker_concurrency=%s "
                    "reranker_batch_size=%s",
                    profile.device,
                    profile.rerank_candidates,
                    profile.rerank_top_n,
                    profile.reranker_concurrency,
                    profile.reranker_batch_size,
                )
                cache = self._create_query_cache()
                bm25_repository = self._create_bm25_repository()
                retriever = HybridRetrievalService(
                    vector_retriever=self._get_vector_retriever(),
                    bm25_retriever=LazyBm25CandidateRetrieverAdapter(
                        repository=bm25_repository,
                        corpus_reader=self._get_chunk_corpus(),
                        top_k=10,
                    ),
                    top_k_per_source=10,
                )
                reranker = RerankerProviderAdapter(
                    device=profile.device,
                    inference_concurrency=profile.reranker_concurrency,
                    batch_size=profile.reranker_batch_size,
                    model_name=self._config.reranker.model,
                    threshold=self._config.reranker.threshold,
                    debug=self._config.reranker.debug,
                )
                self._reranker = reranker
                self._retrieval_service = RetrievalService(
                    cache=cache,
                    retriever=retriever,
                    reranker=reranker,
                    max_content_chars=self._config.tuning.max_content_chars,
                    rerank_candidates=profile.rerank_candidates,
                    rerank_top_n=profile.rerank_top_n,
                )
                self._hybrid_retriever = retriever
                self._cache_invalidator = RagCacheInvalidator(
                    bm25_cache=bm25_repository,
                    query_cache=cache,
                    retriever_provider=self._current_hybrid_retriever,
                )
        return self._retrieval_service

    def get_evaluation_retrieval_service(self):
        """返回只供 eval-runner 使用的无缓存分阶段检索用例。"""
        if self._evaluation_retrieval_service is not None:
            return self._evaluation_retrieval_service

        self.get_retrieval_service()
        with self._lock:
            if self._evaluation_retrieval_service is None:
                from knowledge.rag.evaluation import EvaluationRetrievalService

                if self._hybrid_retriever is None or self._reranker is None:
                    raise RuntimeError("评测检索依赖尚未完成组装")
                profile = self._config.tuning.profile(self._device_provider())
                self._evaluation_retrieval_service = EvaluationRetrievalService(
                    retriever=self._hybrid_retriever,
                    reranker=self._reranker,
                    max_content_chars=self._config.tuning.max_content_chars,
                    rerank_candidates=profile.rerank_candidates,
                    production_rerank_top_n=profile.rerank_top_n,
                    reranker_batch_size=profile.reranker_batch_size,
                    device=profile.device,
                    source_top_k=10,
                    rrf_rank_constant=60,
                )
        return self._evaluation_retrieval_service

    def get_admin_service(self) -> RagAdminService:
        if self._admin_service is not None:
            return self._admin_service
        with self._lock:
            if self._admin_service is None:
                self._admin_service = self._build_admin_service()
        return self._admin_service

    def _build_admin_service(self) -> RagAdminService:
        return RagAdminService(
            cache_invalidator_provider=self._get_cache_invalidator,
            status_provider=self.operations.get_status,
            knowledge_base=ChromaKnowledgeBaseAdapter(chroma_dir=self._chroma_dir),
            chunk_reader=self._get_chunk_corpus(),
        )

    def create_cache_invalidator(self) -> RagCacheInvalidator:
        """创建可由顶层组合根连接到建库提交事件的 RAG 失效处理器。"""
        return RagCacheInvalidator(
            bm25_cache=self._create_bm25_repository(),
            query_cache=self._create_query_cache(),
            retriever_provider=self._current_hybrid_retriever,
        )

    def _create_query_cache(self) -> RedisSemanticCacheAdapter:
        return RedisSemanticCacheAdapter(
            redis_provider=self._redis_resources.get_async,
            embedding_provider=self._embedding_provider,
            ttl=self._config.redis.semantic_cache_ttl,
            threshold=self._config.redis.semantic_cache_threshold,
            min_score=self._config.redis.semantic_cache_min_score,
        )

    def _create_bm25_repository(self) -> RedisBm25Repository:
        return RedisBm25Repository(
            hmac_secret=self._config.redis.bm25_hmac_secret,
            ttl=self._config.redis.bm25_index_ttl,
            redis_provider=self._redis_resources.get_sync,
        )

    def _get_vector_retriever(self):
        if self._vector_retriever is not None:
            return self._vector_retriever
        with self._lock:
            if self._vector_retriever is None:
                from knowledge.rag.infrastructure.retrieval.chroma_retriever import (
                    ChromaCandidateRetriever,
                )

                self._vector_retriever = ChromaCandidateRetriever(
                    chroma_dir=self._chroma_dir,
                    top_k=10,
                    embedding_provider=self._embedding_provider,
                )
        return self._vector_retriever

    def _get_chunk_corpus(self):
        if self._chunk_corpus is not None:
            return self._chunk_corpus
        with self._lock:
            if self._chunk_corpus is None:
                from knowledge.rag.infrastructure.retrieval.chunk_corpus import (
                    ChromaCorpusReaderAdapter,
                    MigratingChunkCorpusAdapter,
                )

                self._chunk_corpus = MigratingChunkCorpusAdapter(
                    store=self._chunk_store_provider(),
                    legacy_source=ChromaCorpusReaderAdapter(
                        chroma_dir=self._chroma_dir,
                    ),
                    lock=self._knowledge_write_lock,
                )
        return self._chunk_corpus

    def _get_cache_invalidator(self) -> RagCacheInvalidator:
        self.get_retrieval_service()
        if self._cache_invalidator is None:
            raise RuntimeError("RAG 缓存失效协调器尚未完成组装")
        return self._cache_invalidator

    def _current_hybrid_retriever(self) -> HybridRetrieverPort | None:
        return self._hybrid_retriever

    async def close(self) -> None:
        """停止 RAG 后台任务并释放模块独占资源。"""
        resources = (
            ("operations", self.operations),
            ("retriever", self._hybrid_retriever),
            ("reranker", self._reranker),
            ("redis", self._redis_resources),
        )
        for name, resource in resources:
            close = getattr(resource, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:
                logger.exception("[RAG] 关闭资源失败 resource=%s", name)
        self._retrieval_service = None
        self._evaluation_retrieval_service = None
        self._admin_service = None
        self._cache_invalidator = None
        self._hybrid_retriever = None
        self._vector_retriever = None
        self._chunk_corpus = None
        self._reranker = None


__all__ = ["LocalRagRuntime"]
