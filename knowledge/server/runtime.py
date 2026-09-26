"""Compose the local Knowledge Runtime from server-owned environment settings."""

from __future__ import annotations

import os
from pathlib import Path

from knowledge.ingestion.config import (
    DoclingRuntimeConfig,
    IngestionConfig,
    IngestionRuntimeConfig,
)
from knowledge.rag.config import (
    RagRuntimeConfig,
    RagTuningConfig,
    RedisRuntimeConfig,
    RerankerRuntimeConfig,
)
from knowledge.runtime import (
    KnowledgeRuntime,
    KnowledgeRuntimeConfig,
    SharedResourceConfig,
    StorageRuntimeConfig,
)
from knowledge.runtime import create_knowledge_runtime as create_local_knowledge_runtime


_APP_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: object) -> object:
    """仅在环境变量缺失时使用默认值；显式非法值交给配置模型拒绝。"""
    return os.environ[name] if name in os.environ else default


def _optional_env(name: str) -> object | None:
    return os.environ[name] if name in os.environ else None


def _env_path(name: str, default: str) -> str:
    raw = os.environ[name] if name in os.environ else default
    if not raw.strip():
        raise ValueError(f"{name} 不能为空")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _APP_ROOT / path
    return str(path.resolve())


def create_knowledge_runtime_config() -> KnowledgeRuntimeConfig:
    """Build the local runtime DTO without importing Agent configuration."""
    max_pdf_pages = _env("DOCLING_MAX_PDF_PAGES", 200)
    hash_record_path = _env_path("HASH_RECORD_PATH", "./chroma_db/file_hashes.json")
    return KnowledgeRuntimeConfig(
        shared=SharedResourceConfig(
            requested_device=_env("RAG_DEVICE", "auto"),
            embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        ),
        rag=RagRuntimeConfig(
            tuning=RagTuningConfig(
                max_content_chars=_env("RAG_MAX_CONTENT_CHARS", 800),
                rerank_candidates=_optional_env("RAG_RERANK_CANDIDATES"),
                cpu_rerank_candidates=_env("RAG_CPU_RERANK_CANDIDATES", 12),
                cuda_rerank_candidates=_env("RAG_CUDA_RERANK_CANDIDATES", 20),
                rerank_top_n=_env("RAG_RERANK_TOP_N", 5),
                reranker_concurrency=_optional_env("RAG_RERANKER_CONCURRENCY"),
                cpu_reranker_concurrency=_env("RAG_CPU_RERANKER_CONCURRENCY", 1),
                cuda_reranker_concurrency=_env("RAG_CUDA_RERANKER_CONCURRENCY", 1),
                reranker_batch_size=_optional_env("RAG_RERANKER_BATCH_SIZE"),
                cpu_reranker_batch_size=_env("RAG_CPU_RERANKER_BATCH_SIZE", 2),
                cuda_reranker_batch_size=_env("RAG_CUDA_RERANKER_BATCH_SIZE", 32),
            ),
            redis=RedisRuntimeConfig(
                url=_env("REDIS_URL", "redis://localhost:6379"),
                max_connections=_env("REDIS_MAX_CONNECTIONS", 20),
                semantic_cache_ttl=_env("SEMANTIC_CACHE_TTL", 3_600),
                semantic_cache_threshold=_env("SEMANTIC_CACHE_THRESHOLD", 0.95),
                semantic_cache_min_score=_env("CACHE_MIN_SCORE", 0.5),
                bm25_index_ttl=_env("BM25_INDEX_TTL", 86_400),
                bm25_hmac_secret=_env("BM25_HMAC_SECRET", ""),
            ),
            reranker=RerankerRuntimeConfig(
                model=_env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
                threshold=_env("RERANKER_THRESHOLD", 0.1),
                debug=_env("RERANKER_DEBUG", False),
            ),
        ),
        storage=StorageRuntimeConfig(
            chroma_db_path=_env_path("CHROMA_DB_PATH", "./chroma_db"),
            chunk_store_path=_env_path(
                "CHUNK_STORE_PATH", "./data/knowledge/chunks.sqlite3"
            ),
        ),
        ingestion=IngestionRuntimeConfig(
            service=IngestionConfig(
                max_pdf_pages=max_pdf_pages,
                batch_size=_env("RAG_INGESTION_BATCH_SIZE", 1_000),
                workers=_env("RAG_INGESTION_WORKERS", 1),
                fail_fast=_env("RAG_INGESTION_FAIL_FAST", True),
            ),
            docling=DoclingRuntimeConfig(
                enabled=_env("DOCLING_ENABLED", False),
                strict_mode=_env("DOCLING_STRICT_MODE", False),
                parser_mode=_env("DOCLING_PARSER_MODE", "auto"),
                base_url=_env("DOCLING_BASE_URL", "http://127.0.0.1:5001"),
                api_key=_env("DOCLING_API_KEY", ""),
                request_timeout=_env("DOCLING_REQUEST_TIMEOUT", 120.0),
                task_timeout=_env("DOCLING_TASK_TIMEOUT", 3_600.0),
                poll_interval=_env("DOCLING_POLL_INTERVAL", 2.0),
                max_pages_per_task=_env("DOCLING_MAX_PAGES_PER_TASK", 3),
                segment_retries=_env("DOCLING_SEGMENT_RETRIES", 1),
                image_bytes_per_page=_env("DOCLING_AUTO_IMAGE_BYTES_PER_PAGE", 307_200),
                min_text_chars_per_page=_env(
                    "DOCLING_AUTO_MIN_TEXT_CHARS_PER_PAGE", 100
                ),
                min_image_area_ratio=_env("DOCLING_AUTO_MIN_IMAGE_AREA_RATIO", 0.15),
                min_table_like_pages_ratio=_env(
                    "DOCLING_AUTO_MIN_TABLE_LIKE_PAGES_RATIO", 0.6
                ),
                min_multi_column_pages_ratio=_env(
                    "DOCLING_AUTO_MIN_MULTI_COLUMN_PAGES_RATIO", 0.4
                ),
                sample_pages=_env("DOCLING_AUTO_SAMPLE_PAGES", 5),
                max_pdf_pages=max_pdf_pages,
                local_min_page_coverage=_env("DOCLING_LOCAL_MIN_PAGE_COVERAGE", 0.65),
                local_min_chars_per_page=_env("DOCLING_LOCAL_MIN_CHARS_PER_PAGE", 80),
                local_max_replacement_ratio=_env(
                    "DOCLING_LOCAL_MAX_REPLACEMENT_RATIO", 0.005
                ),
                local_min_traceable_ratio=_env(
                    "DOCLING_LOCAL_MIN_TRACEABLE_RATIO", 0.95
                ),
            ),
            hash_record_path=hash_record_path,
        ),
    )


def create_knowledge_runtime() -> KnowledgeRuntime:
    """Create the local runtime owned by the Knowledge HTTP server."""
    return create_local_knowledge_runtime(create_knowledge_runtime_config())


__all__ = ["create_knowledge_runtime", "create_knowledge_runtime_config"]
