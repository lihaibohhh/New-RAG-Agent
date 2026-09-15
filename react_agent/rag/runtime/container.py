"""Instance-scoped composition root for RAG services."""
from __future__ import annotations

import logging
import os
import threading

from react_agent.rag.admin import RagAdminService
from react_agent.rag.ingestion import (
    DocumentParsingService,
    IngestionService,
    PdfParserRouter,
)
from react_agent.rag.operations import RagWarmupManager
from react_agent.rag.infrastructure.models import EmbeddingProviderAdapter
from react_agent.rag.ports import HybridRetrieverPort, RagCacheInvalidatorPort
from react_agent.rag.query import RetrievalService
from react_agent.infrastructure.redis import RedisClientManager


logger = logging.getLogger(__name__)


class RagRuntime:
    """Create, connect and own one isolated RAG object graph."""

    def __init__(self) -> None:
        from react_agent.core.config import settings

        self._retrieval_service: RetrievalService | None = None
        self._evaluation_retrieval_service = None
        self._ingestion_service: IngestionService | None = None
        self._admin_service: RagAdminService | None = None
        self._document_parsing_service: DocumentParsingService | None = None
        self._cache_invalidator: RagCacheInvalidatorPort | None = None
        self._hybrid_retriever: HybridRetrieverPort | None = None
        self._vector_retriever = None
        self._chunk_store = None
        self._chunk_corpus = None
        self._embedding_provider: EmbeddingProviderAdapter | None = None
        self._reranker = None
        self._lock = threading.RLock()
        self._knowledge_write_lock = threading.RLock()
        self._redis_resources = RedisClientManager(
            url=settings.redis.REDIS_URL,
            max_connections=settings.redis.REDIS_MAX_CONNECTIONS,
        )
        self.operations = RagWarmupManager(self.get_retrieval_service)

    def get_retrieval_service(self) -> RetrievalService:
        if self._retrieval_service is not None:
            return self._retrieval_service

        with self._lock:
            if self._retrieval_service is None:
                from react_agent.core.config import settings
                from react_agent.rag.infrastructure.cache.semantic_cache import (
                    RedisSemanticCacheAdapter,
                )
                from react_agent.rag.infrastructure.retrieval.bm25_repository import (
                    RedisBm25Repository,
                )
                from react_agent.rag.infrastructure.retrieval.hybrid_retriever import (
                    HybridRetrieverAdapter,
                )
                from react_agent.rag.infrastructure.retrieval.reranker import (
                    RerankerProviderAdapter,
                )
                from react_agent.rag.runtime.device import get_rag_runtime_profile
                from react_agent.rag.runtime.invalidation import RagCacheInvalidator

                profile = get_rag_runtime_profile()
                embedding_provider = self._get_embedding_provider(profile.device)
                logger.info(
                    "[RAG] runtime profile device=%s timeout=%ss "
                    "rerank_candidates=%s rerank_top_n=%s reranker_concurrency=%s",
                    profile.device,
                    profile.timeout,
                    profile.rerank_candidates,
                    profile.rerank_top_n,
                    profile.reranker_concurrency,
                )
                cache = RedisSemanticCacheAdapter(
                    redis_provider=self._redis_resources.get_async,
                    embedding_provider=embedding_provider.get,
                )
                bm25_repository = RedisBm25Repository(
                    hmac_secret=settings.redis.BM25_HMAC_SECRET,
                    ttl=settings.redis.BM25_INDEX_TTL,
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
                )
                self._reranker = reranker
                self._retrieval_service = RetrievalService(
                    cache=cache,
                    retriever=retriever,
                    reranker=reranker,
                    max_content_chars=settings.tools.rag.max_content_chars,
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
                from react_agent.core.config import settings
                from react_agent.rag.evaluation import EvaluationRetrievalService
                from react_agent.rag.runtime.device import get_rag_runtime_profile

                if self._hybrid_retriever is None or self._reranker is None:
                    raise RuntimeError("评测检索依赖尚未完成组装")
                profile = get_rag_runtime_profile()
                self._evaluation_retrieval_service = EvaluationRetrievalService(
                    retriever=self._hybrid_retriever,
                    reranker=self._reranker,
                    max_content_chars=settings.tools.rag.max_content_chars,
                    rerank_candidates=profile.rerank_candidates,
                    production_rerank_top_n=profile.rerank_top_n,
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
                from react_agent.core.config import settings
                from react_agent.rag.infrastructure.parsing.docling_parser import (
                    DoclingServiceAdapter,
                )
                from react_agent.rag.infrastructure.parsing.ingestion_support import (
                    DoclingPreflightAdapter,
                    PdfPageCounterAdapter,
                )
                from react_agent.rag.infrastructure.storage.chroma_vector_writer import (
                    ChromaVectorWriterAdapter,
                )
                from react_agent.rag.infrastructure.storage.composite_writer import (
                    CompositeVectorIndexWriter,
                )
                from react_agent.rag.infrastructure.storage.file_manifest import (
                    JsonIngestionManifestAdapter,
                )

                config = settings.docling
                preflight_client = DoclingServiceAdapter(
                    base_url=config.DOCLING_BASE_URL,
                    api_key=settings.secrets.DOCLING_API_KEY,
                    request_timeout=config.DOCLING_REQUEST_TIMEOUT,
                    task_timeout=config.DOCLING_TASK_TIMEOUT,
                    poll_interval=config.DOCLING_POLL_INTERVAL,
                )
                self._ingestion_service = IngestionService(
                    writer=CompositeVectorIndexWriter(
                        ChromaVectorWriterAdapter(
                            chroma_dir=settings.tools.vector_store.CHROMA_DB_PATH,
                            embedding_provider=self._get_embedding_provider().get,
                        ),
                        self._get_chunk_store(),
                        lock=self._knowledge_write_lock,
                    ),
                    manifest=JsonIngestionManifestAdapter(),
                    page_counter=PdfPageCounterAdapter(),
                    preflight=DoclingPreflightAdapter(
                        client=preflight_client,
                        enabled=config.DOCLING_ENABLED,
                        strict_mode=config.DOCLING_STRICT_MODE,
                    ),
                    document_parser=document_parser,
                    cache_invalidator=cache_invalidator,
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
        from react_agent.core.config import settings
        from react_agent.rag.infrastructure.parsing.basic_document_parser import (
            BasicDocumentParserAdapter,
        )
        from react_agent.rag.infrastructure.parsing.docling_parser import (
            DoclingServiceAdapter,
        )
        from react_agent.rag.infrastructure.parsing.local_pdf_parser import (
            LocalPdfParserAdapter,
        )
        from react_agent.rag.infrastructure.parsing.structured_table_parser import (
            StructuredTableParserAdapter,
        )

        config = settings.docling
        pdf_parser = PdfParserRouter(
            local_parser=LocalPdfParserAdapter(),
            docling_parser=DoclingServiceAdapter(
                base_url=config.DOCLING_BASE_URL,
                api_key=settings.secrets.DOCLING_API_KEY,
                request_timeout=config.DOCLING_REQUEST_TIMEOUT,
                task_timeout=config.DOCLING_TASK_TIMEOUT,
                poll_interval=config.DOCLING_POLL_INTERVAL,
                max_pages_per_task=config.DOCLING_MAX_PAGES_PER_TASK,
                segment_retries=config.DOCLING_SEGMENT_RETRIES,
            ),
            docling_enabled=config.DOCLING_ENABLED,
            strict_mode=config.DOCLING_STRICT_MODE,
            mode=config.DOCLING_PARSER_MODE,
            image_bytes_per_page=config.DOCLING_AUTO_IMAGE_BYTES_PER_PAGE,
            min_text_chars_per_page=config.DOCLING_AUTO_MIN_TEXT_CHARS_PER_PAGE,
            min_image_area_ratio=config.DOCLING_AUTO_MIN_IMAGE_AREA_RATIO,
            min_table_like_pages_ratio=config.DOCLING_AUTO_MIN_TABLE_LIKE_PAGES_RATIO,
            min_multi_column_pages_ratio=config.DOCLING_AUTO_MIN_MULTI_COLUMN_PAGES_RATIO,
            max_pdf_pages=config.DOCLING_MAX_PDF_PAGES,
            local_min_page_coverage=config.DOCLING_LOCAL_MIN_PAGE_COVERAGE,
            local_min_chars_per_page=config.DOCLING_LOCAL_MIN_CHARS_PER_PAGE,
            local_max_replacement_ratio=config.DOCLING_LOCAL_MAX_REPLACEMENT_RATIO,
            local_min_traceable_ratio=config.DOCLING_LOCAL_MIN_TRACEABLE_RATIO,
            sample_pages=config.DOCLING_AUTO_SAMPLE_PAGES,
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
        from react_agent.rag.infrastructure.storage.chroma_knowledge_base import (
            ChromaKnowledgeBaseAdapter,
        )

        return RagAdminService(
            cache_invalidator_provider=self._get_cache_invalidator,
            status_provider=self.operations.get_status,
            knowledge_base=ChromaKnowledgeBaseAdapter(chroma_dir=chroma_dir),
            chunk_reader=(self._get_chunk_corpus() if chroma_dir is None else None),
        )

    def _get_vector_retriever(self):
        if self._vector_retriever is not None:
            return self._vector_retriever
        with self._lock:
            if self._vector_retriever is None:
                from react_agent.core.config import settings
                from react_agent.rag.infrastructure.retrieval.chroma_retriever import (
                    ChromaCandidateRetriever,
                )

                self._vector_retriever = ChromaCandidateRetriever(
                    chroma_dir=settings.tools.vector_store.CHROMA_DB_PATH,
                    top_k=10,
                    embedding_provider=self._get_embedding_provider().get,
                )
        return self._vector_retriever

    def _get_chunk_store(self):
        if self._chunk_store is not None:
            return self._chunk_store
        with self._lock:
            if self._chunk_store is None:
                from react_agent.core.config import settings
                from react_agent.rag.infrastructure.storage.sqlite_chunk_store import (
                    SQLiteChunkStoreAdapter,
                )

                self._chunk_store = SQLiteChunkStoreAdapter(
                    database_path=settings.tools.vector_store.CHUNK_STORE_PATH,
                )
        return self._chunk_store

    def _get_chunk_corpus(self):
        if self._chunk_corpus is not None:
            return self._chunk_corpus
        with self._lock:
            if self._chunk_corpus is None:
                from react_agent.rag.infrastructure.retrieval.chunk_corpus import (
                    ChromaCorpusReaderAdapter,
                    MigratingChunkCorpusAdapter,
                )
                from react_agent.core.config import settings

                self._chunk_corpus = MigratingChunkCorpusAdapter(
                    store=self._get_chunk_store(),
                    legacy_source=ChromaCorpusReaderAdapter(
                        chroma_dir=settings.tools.vector_store.CHROMA_DB_PATH,
                    ),
                    lock=self._knowledge_write_lock,
                )
        return self._chunk_corpus

    def _get_cache_invalidator(self) -> RagCacheInvalidatorPort:
        self.get_retrieval_service()
        if self._cache_invalidator is None:
            raise RuntimeError("RAG 缓存失效协调器尚未完成组装")
        return self._cache_invalidator

    def _current_hybrid_retriever(self) -> HybridRetrieverPort | None:
        return self._hybrid_retriever

    def _create_cache_invalidator(self) -> RagCacheInvalidatorPort:
        from react_agent.core.config import settings
        from react_agent.rag.infrastructure.cache.semantic_cache import (
            RedisSemanticCacheAdapter,
        )
        from react_agent.rag.infrastructure.retrieval.bm25_repository import (
            RedisBm25Repository,
        )
        from react_agent.rag.runtime.invalidation import RagCacheInvalidator

        return RagCacheInvalidator(
            bm25_cache=RedisBm25Repository(
                hmac_secret=settings.redis.BM25_HMAC_SECRET,
                ttl=settings.redis.BM25_INDEX_TTL,
                redis_provider=self._redis_resources.get_sync,
            ),
            query_cache=RedisSemanticCacheAdapter(
                redis_provider=self._redis_resources.get_async,
                embedding_provider=self._get_embedding_provider().get,
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
                    from react_agent.rag.runtime.device import get_rag_runtime_profile

                    device = get_rag_runtime_profile().device
                self._embedding_provider = EmbeddingProviderAdapter(
                    model_name=os.getenv(
                        "EMBEDDING_MODEL",
                        "BAAI/bge-small-zh-v1.5",
                    ),
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


def create_rag_runtime() -> RagRuntime:
    """Create a new isolated RAG composition scope."""
    return RagRuntime()


__all__ = ["RagRuntime", "create_rag_runtime"]
