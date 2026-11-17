"""RAG 模块拥有的本地对象图与生命周期。"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, Literal

from knowledge.contracts import (
    EvaluationRetrievalResult,
    RagHealthStatus,
    StoredChunk,
)
from knowledge.rag.admin import RagAdminService
from knowledge.rag.application import RetrievalApplicationService
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
from knowledge.rag.infrastructure.retrieval.chroma_retriever import (
    ChromaCandidateRetriever,
)
from knowledge.rag.infrastructure.retrieval.chunk_corpus import (
    ChromaCorpusReaderAdapter,
    MigratingChunkCorpusAdapter,
)
from knowledge.rag.infrastructure.retrieval.reranker import (
    RerankerProviderAdapter,
)
from knowledge.rag.infrastructure.storage import ChromaKnowledgeBaseAdapter
from knowledge.settings import RagSettings
from knowledge.rag.operations import RagWarmupManager
from knowledge.rag.ports import ChunkStorePort
from knowledge.rag.retrieval import HybridRetrievalService

logger = logging.getLogger(__name__)


class LocalRagService(RetrievalApplicationService):
    """可直接检索的应用服务，同时管理评测、运维和资源生命周期。"""

    def __init__(
        self,
        config: RagSettings,
        *,
        chroma_dir: str,
        device_provider: Callable[[], Literal["cpu", "cuda"]],
        embedding_provider: Callable[[], Any],
        chunk_store_provider: Callable[[], ChunkStorePort],
        knowledge_write_lock: threading.RLock,
    ) -> None:
        profile = config.tuning.profile(device_provider())
        logger.info(
            "[RAG] runtime profile device=%s rerank_candidates=%s "
            "rerank_top_n=%s reranker_concurrency=%s reranker_batch_size=%s",
            profile.device,
            profile.rerank_candidates,
            profile.rerank_top_n,
            profile.reranker_concurrency,
            profile.reranker_batch_size,
        )
        self._config = config
        self._profile = profile
        self._lock = threading.RLock()
        self._redis_resources = RedisClientManager(
            url=config.redis.url,
            max_connections=config.redis.max_connections,
        )
        query_cache = RedisSemanticCacheAdapter(
            redis_provider=self._redis_resources.get_async,
            embedding_provider=embedding_provider,
            ttl=config.redis.semantic_cache_ttl,
            threshold=config.redis.semantic_cache_threshold,
            min_score=config.redis.semantic_cache_min_score,
        )
        bm25_repository = RedisBm25Repository(
            hmac_secret=config.redis.bm25_hmac_secret,
            ttl=config.redis.bm25_index_ttl,
            redis_provider=self._redis_resources.get_sync,
        )
        self._chunk_corpus = MigratingChunkCorpusAdapter(
            store=chunk_store_provider(),
            legacy_source=ChromaCorpusReaderAdapter(chroma_dir=chroma_dir),
            lock=knowledge_write_lock,
        )
        self._vector_retriever = ChromaCandidateRetriever(
            chroma_dir=chroma_dir,
            top_k=10,
            embedding_provider=embedding_provider,
        )
        self._hybrid_retriever = HybridRetrievalService(
            vector_retriever=self._vector_retriever,
            bm25_retriever=LazyBm25CandidateRetrieverAdapter(
                repository=bm25_repository,
                corpus_reader=self._chunk_corpus,
                top_k=10,
            ),
            top_k_per_source=10,
        )
        self._reranker = RerankerProviderAdapter(
            device=profile.device,
            inference_concurrency=profile.reranker_concurrency,
            batch_size=profile.reranker_batch_size,
            model_name=config.reranker.model,
            threshold=config.reranker.threshold,
            debug=config.reranker.debug,
        )
        super().__init__(
            cache=query_cache,
            retriever=self._hybrid_retriever,
            reranker=self._reranker,
            max_content_chars=config.tuning.max_content_chars,
            rerank_candidates=profile.rerank_candidates,
            rerank_top_n=profile.rerank_top_n,
        )
        self._cache_invalidator = RagCacheInvalidator(
            bm25_cache=bm25_repository,
            query_cache=query_cache,
            retriever_provider=lambda: self._hybrid_retriever,
        )
        self._operations = RagWarmupManager(lambda: self)
        self._admin_service = RagAdminService(
            cache_invalidator_provider=lambda: self._cache_invalidator,
            status_provider=self._operations.get_status,
            knowledge_base=ChromaKnowledgeBaseAdapter(chroma_dir=chroma_dir),
            chunk_reader=self._chunk_corpus,
        )
        self._evaluation_retrieval_service = None

    def _get_evaluation_retrieval_service(self):
        """返回只供 eval-runner 使用的无缓存分阶段检索用例。"""
        if self._evaluation_retrieval_service is not None:
            return self._evaluation_retrieval_service

        with self._lock:
            if self._evaluation_retrieval_service is None:
                from knowledge.rag.evaluation import EvaluationRetrievalService

                self._evaluation_retrieval_service = EvaluationRetrievalService(
                    retriever=self._hybrid_retriever,
                    reranker=self._reranker,
                    max_content_chars=self._config.tuning.max_content_chars,
                    rerank_candidates=self._profile.rerank_candidates,
                    production_rerank_top_n=self._profile.rerank_top_n,
                    reranker_batch_size=self._profile.reranker_batch_size,
                    device=self._profile.device,
                    source_top_k=10,
                    rrf_rank_constant=60,
                )
        return self._evaluation_retrieval_service

    def create_index_changed_handler(self) -> RagCacheInvalidator:
        """向建库模块公开唯一的索引变更处理器。"""
        return self._cache_invalidator

    async def evaluation_search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        retrieval_mode: str = "hybrid",
        use_query_cache: bool = False,
    ) -> EvaluationRetrievalResult:
        if use_query_cache:
            raise ValueError("评测检索管道不允许使用语义查询缓存")
        return await self._get_evaluation_retrieval_service().search(
            query,
            top_k=top_k,
            filters=filters,
            retrieval_mode=retrieval_mode,
        )

    async def start_warmup(self, force: bool = False) -> dict[str, Any]:
        return self._operations.start(force=force)

    async def get_warmup_status(self) -> dict[str, Any]:
        return self._operations.get_status()

    async def ensure_ready(
        self,
        wait_seconds: int | float = 20,
    ) -> dict[str, Any]:
        return await self._operations.ensure_ready(wait_seconds)

    async def health(self) -> RagHealthStatus:
        return await self._admin_service.health()

    async def invalidate(self, knowledge_base_id: str = "default") -> None:
        await self._admin_service.invalidate(knowledge_base_id)

    async def read_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredChunk, ...]:
        return await self._admin_service.read_chunks(
            source_file=source_file,
            offset=offset,
            limit=limit,
        )

    def read_chunks_sync(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredChunk, ...]:
        return self._admin_service.read_chunks_sync(
            source_file=source_file,
            offset=offset,
            limit=limit,
        )

    async def close(self) -> None:
        """停止 RAG 后台任务并释放模块独占资源。"""
        resources = (
            ("operations", self._operations),
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
        self._evaluation_retrieval_service = None


__all__ = ["LocalRagService"]
