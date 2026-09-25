"""Instance-scoped composition root for RAG services."""
from __future__ import annotations

import logging
import threading

from knowledge.admin import RagAdminService
from knowledge.ingestion import (
    DocumentParsingService,
    IngestionService,
    PdfParserRouter,
)
from knowledge.infrastructure.models import EmbeddingProviderAdapter
from knowledge.infrastructure.redis import RedisClientManager
from knowledge.rag.operations import RagWarmupManager
from knowledge.ports import CacheInvalidationCoordinatorPort, HybridRetrieverPort
from knowledge.rag.query import RetrievalService
from knowledge.runtime.config import KnowledgeRuntimeConfig


logger = logging.getLogger(__name__)


class KnowledgeRuntime:
    """Knowledge Server 的本地组合根，持有 RAG 与建库共享资源。"""

    def __init__(self, config: KnowledgeRuntimeConfig) -> None:
        self._config = config
        self._retrieval_service: RetrievalService | None = None
        self._evaluation_retrieval_service = None
        self._ingestion_service: IngestionService | None = None
        self._admin_service: RagAdminService | None = None
        self._document_parsing_service: DocumentParsingService | None = None
        self._cache_invalidator: CacheInvalidationCoordinatorPort | None = None
        self._hybrid_retriever: HybridRetrieverPort | None = None
        self._vector_retriever = None
        self._chunk_store = None
        self._chunk_corpus = None
        self._embedding_provider: EmbeddingProviderAdapter | None = None
        self._reranker = None
        self._lock = threading.RLock()
        self._knowledge_write_lock = threading.RLock()
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
                from knowledge.infrastructure.cache.semantic_cache import (
                    RedisSemanticCacheAdapter,
                )
                from knowledge.infrastructure.retrieval.bm25_repository import (
                    RedisBm25Repository,
                )
                from knowledge.infrastructure.retrieval.hybrid_retriever import (
                    HybridRetrieverAdapter,
                )
                from knowledge.infrastructure.retrieval.reranker import (
                    RerankerProviderAdapter,
                )
                from knowledge.runtime.device import get_rag_runtime_profile
                from knowledge.runtime.invalidation import RagCacheInvalidator

                profile = get_rag_runtime_profile(self._config)
                embedding_provider = self._get_embedding_provider(profile.device)
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
                cache = RedisSemanticCacheAdapter(
                    redis_provider=self._redis_resources.get_async,
                    embedding_provider=embedding_provider.get,
                    ttl=self._config.redis.semantic_cache_ttl,
                    threshold=self._config.redis.semantic_cache_threshold,
                    min_score=self._config.redis.semantic_cache_min_score,
                )
                bm25_repository = RedisBm25Repository(
                    hmac_secret=self._config.redis.bm25_hmac_secret,
                    ttl=self._config.redis.bm25_index_ttl,
                    redis_provider=self._redis_resources.get_sync,
                )
                vector_retriever = self._get_vector_retriever()
                retriever = HybridRetrieverAdapter(
                    vector_retriever=vector_retriever,
                    bm25_repository=bm25_repository,
                    corpus_reader=self._get_chunk_corpus(),
                    top_k_per_source=10,
                )
                reranker = RerankerProviderAdapter(
                    device=profile.device,
                    inference_concurrency=profile.reranker_concurrency,
                    batch_size=profile.reranker_batch_size,
                    model_name=self._config.models.reranker_model,
                    threshold=self._config.models.reranker_threshold,
                    debug=self._config.models.reranker_debug,
                )
                self._reranker = reranker
                self._retrieval_service = RetrievalService(
                    cache=cache,
                    retriever=retriever,
                    reranker=reranker,
                    max_content_chars=self._config.rag.max_content_chars,
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
                from knowledge.runtime.device import get_rag_runtime_profile

                if self._hybrid_retriever is None or self._reranker is None:
                    raise RuntimeError("评测检索依赖尚未完成组装")
                profile = get_rag_runtime_profile(self._config)
                self._evaluation_retrieval_service = EvaluationRetrievalService(
                    retriever=self._hybrid_retriever,
                    reranker=self._reranker,
                    max_content_chars=self._config.rag.max_content_chars,
                    rerank_candidates=profile.rerank_candidates,
                    production_rerank_top_n=profile.rerank_top_n,
                    reranker_batch_size=profile.reranker_batch_size,
                    device=profile.device,
                    source_top_k=10,
                    rrf_rank_constant=60,
                )
        return self._evaluation_retrieval_service

    def get_ingestion_service(self) -> IngestionService:
        if self._ingestion_service is not None:
            return self._ingestion_service

        document_parser = self.get_document_parsing_service()
        cache_invalidator = self._create_cache_invalidator()

        with self._lock:
            if self._ingestion_service is None:
                from knowledge.infrastructure.parsing.docling_parser import (
                    DoclingServiceAdapter,
                )
                from knowledge.infrastructure.parsing.ingestion_support import (
                    DoclingPreflightAdapter,
                    PdfPageCounterAdapter,
                )
                from knowledge.infrastructure.storage.chroma_vector_writer import (
                    ChromaVectorWriterAdapter,
                )
                from knowledge.infrastructure.storage.composite_writer import (
                    CompositeVectorIndexWriter,
                )
                from knowledge.infrastructure.storage.file_manifest import (
                    JsonIngestionManifestAdapter,
                )

                docling_config = self._config.docling
                preflight_client = DoclingServiceAdapter(
                    base_url=docling_config.base_url,
                    api_key=docling_config.api_key,
                    request_timeout=docling_config.request_timeout,
                    task_timeout=docling_config.task_timeout,
                    poll_interval=docling_config.poll_interval,
                )
                self._ingestion_service = IngestionService(
                    writer=CompositeVectorIndexWriter(
                        ChromaVectorWriterAdapter(
                            chroma_dir=self._config.storage.chroma_db_path,
                            embedding_provider=self._get_embedding_provider().get,
                        ),
                        self._get_chunk_store(),
                        lock=self._knowledge_write_lock,
                    ),
                    manifest=JsonIngestionManifestAdapter(
                        self._config.storage.hash_record_path
                    ),
                    page_counter=PdfPageCounterAdapter(),
                    preflight=DoclingPreflightAdapter(
                        client=preflight_client,
                        enabled=docling_config.enabled,
                        strict_mode=docling_config.strict_mode,
                    ),
                    document_parser=document_parser,
                    index_changed=cache_invalidator,
                    config=self._config.ingestion,
                )
        return self._ingestion_service

    def get_document_parsing_service(self) -> DocumentParsingService:
        if self._document_parsing_service is not None:
            return self._document_parsing_service
        with self._lock:
            if self._document_parsing_service is None:
                self._document_parsing_service = self._build_document_parsing_service()
        return self._document_parsing_service

    def _build_document_parsing_service(self) -> DocumentParsingService:
        from knowledge.infrastructure.parsing.basic_document_parser import (
            BasicDocumentParserAdapter,
        )
        from knowledge.infrastructure.parsing.docling_parser import (
            DoclingServiceAdapter,
        )
        from knowledge.infrastructure.parsing.local_pdf_parser import (
            LocalPdfParserAdapter,
        )
        from knowledge.infrastructure.parsing.structured_table_parser import (
            StructuredTableParserAdapter,
        )

        config = self._config.docling
        pdf_parser = PdfParserRouter(
            local_parser=LocalPdfParserAdapter(),
            docling_parser=DoclingServiceAdapter(
                base_url=config.base_url,
                api_key=config.api_key,
                request_timeout=config.request_timeout,
                task_timeout=config.task_timeout,
                poll_interval=config.poll_interval,
                max_pages_per_task=config.max_pages_per_task,
                segment_retries=config.segment_retries,
            ),
            docling_enabled=config.enabled,
            strict_mode=config.strict_mode,
            mode=config.parser_mode,
            image_bytes_per_page=config.image_bytes_per_page,
            min_text_chars_per_page=config.min_text_chars_per_page,
            min_image_area_ratio=config.min_image_area_ratio,
            min_table_like_pages_ratio=config.min_table_like_pages_ratio,
            min_multi_column_pages_ratio=config.min_multi_column_pages_ratio,
            max_pdf_pages=config.max_pdf_pages,
            local_min_page_coverage=config.local_min_page_coverage,
            local_min_chars_per_page=config.local_min_chars_per_page,
            local_max_replacement_ratio=config.local_max_replacement_ratio,
            local_min_traceable_ratio=config.local_min_traceable_ratio,
            sample_pages=config.sample_pages,
        )
        return DocumentParsingService(
            pdf_parser=pdf_parser,
            basic_parser=BasicDocumentParserAdapter(),
            table_parser=StructuredTableParserAdapter(),
        )

    def get_admin_service(self) -> RagAdminService:
        if self._admin_service is not None:
            return self._admin_service
        with self._lock:
            if self._admin_service is None:
                self._admin_service = self._build_admin_service()
        return self._admin_service

    def create_admin_service(
        self,
        *,
        chroma_dir: str | None = None,
    ) -> RagAdminService:
        return self._build_admin_service(chroma_dir=chroma_dir)

    def _build_admin_service(
        self,
        *,
        chroma_dir: str | None = None,
    ) -> RagAdminService:
        from knowledge.infrastructure.storage.chroma_knowledge_base import (
            ChromaKnowledgeBaseAdapter,
        )

        return RagAdminService(
            cache_invalidator_provider=self._get_cache_invalidator,
            status_provider=self.operations.get_status,
            knowledge_base=ChromaKnowledgeBaseAdapter(
                chroma_dir=chroma_dir or self._config.storage.chroma_db_path
            ),
            chunk_reader=(self._get_chunk_corpus() if chroma_dir is None else None),
        )

    def _get_vector_retriever(self):
        if self._vector_retriever is not None:
            return self._vector_retriever
        with self._lock:
            if self._vector_retriever is None:
                from knowledge.infrastructure.retrieval.chroma_retriever import (
                    ChromaCandidateRetriever,
                )

                self._vector_retriever = ChromaCandidateRetriever(
                    chroma_dir=self._config.storage.chroma_db_path,
                    top_k=10,
                    embedding_provider=self._get_embedding_provider().get,
                )
        return self._vector_retriever

    def _get_chunk_store(self):
        if self._chunk_store is not None:
            return self._chunk_store
        with self._lock:
            if self._chunk_store is None:
                from knowledge.infrastructure.storage.sqlite_chunk_store import (
                    SQLiteChunkStoreAdapter,
                )

                self._chunk_store = SQLiteChunkStoreAdapter(
                    database_path=self._config.storage.chunk_store_path,
                )
        return self._chunk_store

    def _get_chunk_corpus(self):
        if self._chunk_corpus is not None:
            return self._chunk_corpus
        with self._lock:
            if self._chunk_corpus is None:
                from knowledge.infrastructure.retrieval.chunk_corpus import (
                    ChromaCorpusReaderAdapter,
                    MigratingChunkCorpusAdapter,
                )

                self._chunk_corpus = MigratingChunkCorpusAdapter(
                    store=self._get_chunk_store(),
                    legacy_source=ChromaCorpusReaderAdapter(
                        chroma_dir=self._config.storage.chroma_db_path,
                    ),
                    lock=self._knowledge_write_lock,
                )
        return self._chunk_corpus

    def _get_cache_invalidator(self) -> CacheInvalidationCoordinatorPort:
        self.get_retrieval_service()
        if self._cache_invalidator is None:
            raise RuntimeError("RAG 缓存失效协调器尚未完成组装")
        return self._cache_invalidator

    def _current_hybrid_retriever(self) -> HybridRetrieverPort | None:
        return self._hybrid_retriever

    def _create_cache_invalidator(self) -> CacheInvalidationCoordinatorPort:
        from knowledge.infrastructure.cache.semantic_cache import (
            RedisSemanticCacheAdapter,
        )
        from knowledge.infrastructure.retrieval.bm25_repository import (
            RedisBm25Repository,
        )
        from knowledge.runtime.invalidation import RagCacheInvalidator

        return RagCacheInvalidator(
            bm25_cache=RedisBm25Repository(
                hmac_secret=self._config.redis.bm25_hmac_secret,
                ttl=self._config.redis.bm25_index_ttl,
                redis_provider=self._redis_resources.get_sync,
            ),
            query_cache=RedisSemanticCacheAdapter(
                redis_provider=self._redis_resources.get_async,
                embedding_provider=self._get_embedding_provider().get,
                ttl=self._config.redis.semantic_cache_ttl,
                threshold=self._config.redis.semantic_cache_threshold,
                min_score=self._config.redis.semantic_cache_min_score,
            ),
            retriever_provider=self._current_hybrid_retriever,
        )

    def _get_embedding_provider(
        self,
        device: str | None = None,
    ) -> EmbeddingProviderAdapter:
        if self._embedding_provider is not None:
            return self._embedding_provider
        with self._lock:
            if self._embedding_provider is None:
                if device is None:
                    from knowledge.runtime.device import get_rag_runtime_profile

                    device = get_rag_runtime_profile(self._config).device
                self._embedding_provider = EmbeddingProviderAdapter(
                    model_name=self._config.models.embedding_model,
                    device=device,
                )
        return self._embedding_provider

    async def close(self) -> None:
        """Stop operational tasks and release resources owned by this graph."""
        resources = (
            ("operations", self.operations),
            ("retriever", self._hybrid_retriever),
            ("vector_retriever", self._vector_retriever),
            ("reranker", self._reranker),
            ("embedding", self._embedding_provider),
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
        self._ingestion_service = None
        self._admin_service = None
        self._document_parsing_service = None
        self._cache_invalidator = None
        self._hybrid_retriever = None
        self._vector_retriever = None
        self._chunk_store = None
        self._chunk_corpus = None
        self._embedding_provider = None
        self._reranker = None


def create_knowledge_runtime(config: KnowledgeRuntimeConfig) -> KnowledgeRuntime:
    """创建由 Knowledge Server 独占的本地组合范围。"""
    return KnowledgeRuntime(config)


__all__ = ["KnowledgeRuntime", "create_knowledge_runtime"]
