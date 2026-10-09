"""Knowledge 建库、检索、存储与服务边界的统一配置。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _FrozenSettings(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class RagProfile(_FrozenSettings):
    device: Literal["cpu", "cuda"]
    rerank_candidates: int = Field(ge=1, le=100)
    rerank_top_n: int = Field(ge=1, le=20)
    reranker_concurrency: int = Field(ge=1, le=8)
    reranker_batch_size: int = Field(ge=1, le=128)


class RagTuningSettings(_FrozenSettings):
    max_content_chars: int = Field(default=800, ge=200, le=5_000)
    rerank_candidates: int | None = Field(default=None, ge=1, le=100)
    cpu_rerank_candidates: int = Field(default=12, ge=1, le=100)
    cuda_rerank_candidates: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    reranker_concurrency: int | None = Field(default=None, ge=1, le=8)
    cpu_reranker_concurrency: int = Field(default=1, ge=1, le=8)
    cuda_reranker_concurrency: int = Field(default=1, ge=1, le=8)
    reranker_batch_size: int | None = Field(default=None, ge=1, le=128)
    cpu_reranker_batch_size: int = Field(default=2, ge=1, le=128)
    cuda_reranker_batch_size: int = Field(default=32, ge=1, le=128)

    def profile(self, device: Literal["cpu", "cuda"]) -> RagProfile:
        is_cuda = device == "cuda"
        return RagProfile(
            device=device,
            rerank_candidates=self.rerank_candidates
            or (self.cuda_rerank_candidates if is_cuda else self.cpu_rerank_candidates),
            rerank_top_n=self.rerank_top_n,
            reranker_concurrency=self.reranker_concurrency
            or (
                self.cuda_reranker_concurrency
                if is_cuda
                else self.cpu_reranker_concurrency
            ),
            reranker_batch_size=self.reranker_batch_size
            or (
                self.cuda_reranker_batch_size
                if is_cuda
                else self.cpu_reranker_batch_size
            ),
        )


class RedisSettings(_FrozenSettings):
    url: str = Field(default="redis://localhost:6379", min_length=1)
    max_connections: int = Field(default=20, gt=0)
    semantic_cache_ttl: int = Field(default=3_600, gt=0)
    semantic_cache_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    semantic_cache_min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_index_ttl: int = Field(default=86_400, gt=0)
    bm25_hmac_secret: str = ""


class RerankerSettings(_FrozenSettings):
    model: str = Field(default="BAAI/bge-reranker-v2-m3", min_length=1)
    threshold: float = Field(default=0.1, ge=0.0, le=1.0, allow_inf_nan=False)
    debug: bool = False


class RagSettings(_FrozenSettings):
    """在线 RAG 检索的全部参数。"""

    tuning: RagTuningSettings = Field(default_factory=RagTuningSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)


class IngestionBatchSettings(_FrozenSettings):
    max_pdf_pages: int = Field(default=200, gt=0)
    batch_size: int = Field(default=1_000, ge=1, le=20_000)
    workers: int = Field(default=1, ge=1, le=16)
    fail_fast: bool = True


class DoclingSettings(_FrozenSettings):
    enabled: bool = False
    strict_mode: bool = False
    parser_mode: Literal["auto", "local", "docling"] = "auto"
    base_url: str = Field(default="http://127.0.0.1:5001", min_length=1)
    api_key: str = ""
    request_timeout: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    task_timeout: float = Field(default=3_600.0, gt=0, allow_inf_nan=False)
    poll_interval: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    max_pages_per_task: int = Field(default=3, gt=0)
    segment_retries: int = Field(default=1, ge=0)
    image_bytes_per_page: int = Field(default=307_200, gt=0)
    min_text_chars_per_page: int = Field(default=100, ge=0)
    min_image_area_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    min_table_like_pages_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    min_multi_column_pages_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    sample_pages: int = Field(default=5, gt=0)
    max_pdf_pages: int = Field(default=200, gt=0)
    local_min_page_coverage: float = Field(default=0.65, ge=0.0, le=1.0)
    local_min_chars_per_page: int = Field(default=80, ge=0)
    local_max_replacement_ratio: float = Field(default=0.005, ge=0.0, le=1.0)
    local_min_traceable_ratio: float = Field(default=0.95, ge=0.0, le=1.0)

    @field_validator("parser_mode", mode="before")
    @classmethod
    def _normalize_parser_mode(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class IngestionSettings(_FrozenSettings):
    """文档解析与增量建库的全部参数。"""

    batch: IngestionBatchSettings = Field(default_factory=IngestionBatchSettings)
    docling: DoclingSettings = Field(default_factory=DoclingSettings)
    hash_record_path: str = Field(
        default="./chroma_db/file_hashes.json",
        min_length=1,
    )


class StorageSettings(_FrozenSettings):
    chroma_db_path: str = Field(default="./chroma_db", min_length=1)
    chunk_store_path: str = Field(
        default="./data/knowledge/chunks.sqlite3",
        min_length=1,
    )


class SharedSettings(_FrozenSettings):
    requested_device: Literal["auto", "cpu", "cuda"] = "auto"
    embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", min_length=1)

    @field_validator("requested_device", mode="before")
    @classmethod
    def _normalize_device(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class KnowledgeServiceSettings(_FrozenSettings):
    api_key: str = ""
    require_api_key: bool = False
    ingestion_root: Path = Path("./FinancialResearchReportData")
    warmup_on_start: bool = False
    evaluation_api_enabled: bool = False


class KnowledgeClientSettings(_FrozenSettings):
    base_url: str = Field(min_length=1)
    api_key: str = ""
    timeout: float = Field(default=150.0, ge=1.0, allow_inf_nan=False)

    @field_validator("base_url", mode="before")
    @classmethod
    def _normalize_base_url(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().rstrip("/")
        return value


class KnowledgeSettings(_FrozenSettings):
    """Knowledge Service 唯一顶层配置。"""

    shared: SharedSettings = Field(default_factory=SharedSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    service: KnowledgeServiceSettings = Field(default_factory=KnowledgeServiceSettings)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "KnowledgeSettings":
        source = os.environ if environ is None else environ
        max_pdf_pages = _env(source, "DOCLING_MAX_PDF_PAGES", 200)
        return cls(
            shared=SharedSettings(
                requested_device=_env(source, "RAG_DEVICE", "auto"),
                embedding_model=_env(
                    source,
                    "EMBEDDING_MODEL",
                    "BAAI/bge-small-zh-v1.5",
                ),
            ),
            storage=StorageSettings(
                chroma_db_path=_env_path(
                    source,
                    "CHROMA_DB_PATH",
                    "./chroma_db",
                ),
                chunk_store_path=_env_path(
                    source,
                    "CHUNK_STORE_PATH",
                    "./data/knowledge/chunks.sqlite3",
                ),
            ),
            rag=RagSettings(
                tuning=RagTuningSettings(
                    max_content_chars=_env(source, "RAG_MAX_CONTENT_CHARS", 800),
                    rerank_candidates=_optional_env(source, "RAG_RERANK_CANDIDATES"),
                    cpu_rerank_candidates=_env(
                        source,
                        "RAG_CPU_RERANK_CANDIDATES",
                        12,
                    ),
                    cuda_rerank_candidates=_env(
                        source,
                        "RAG_CUDA_RERANK_CANDIDATES",
                        20,
                    ),
                    rerank_top_n=_env(source, "RAG_RERANK_TOP_N", 5),
                    reranker_concurrency=_optional_env(
                        source,
                        "RAG_RERANKER_CONCURRENCY",
                    ),
                    cpu_reranker_concurrency=_env(
                        source,
                        "RAG_CPU_RERANKER_CONCURRENCY",
                        1,
                    ),
                    cuda_reranker_concurrency=_env(
                        source,
                        "RAG_CUDA_RERANKER_CONCURRENCY",
                        1,
                    ),
                    reranker_batch_size=_optional_env(
                        source,
                        "RAG_RERANKER_BATCH_SIZE",
                    ),
                    cpu_reranker_batch_size=_env(
                        source,
                        "RAG_CPU_RERANKER_BATCH_SIZE",
                        2,
                    ),
                    cuda_reranker_batch_size=_env(
                        source,
                        "RAG_CUDA_RERANKER_BATCH_SIZE",
                        32,
                    ),
                ),
                redis=RedisSettings(
                    url=_env(source, "REDIS_URL", "redis://localhost:6379"),
                    max_connections=_env(source, "REDIS_MAX_CONNECTIONS", 20),
                    semantic_cache_ttl=_env(source, "SEMANTIC_CACHE_TTL", 3_600),
                    semantic_cache_threshold=_env(
                        source,
                        "SEMANTIC_CACHE_THRESHOLD",
                        0.95,
                    ),
                    semantic_cache_min_score=_env(source, "CACHE_MIN_SCORE", 0.5),
                    bm25_index_ttl=_env(source, "BM25_INDEX_TTL", 86_400),
                    bm25_hmac_secret=_env(source, "BM25_HMAC_SECRET", ""),
                ),
                reranker=RerankerSettings(
                    model=_env(
                        source,
                        "RERANKER_MODEL",
                        "BAAI/bge-reranker-v2-m3",
                    ),
                    threshold=_env(source, "RERANKER_THRESHOLD", 0.1),
                    debug=_env(source, "RERANKER_DEBUG", False),
                ),
            ),
            ingestion=IngestionSettings(
                batch=IngestionBatchSettings(
                    max_pdf_pages=max_pdf_pages,
                    batch_size=_env(source, "RAG_INGESTION_BATCH_SIZE", 1_000),
                    workers=_env(source, "RAG_INGESTION_WORKERS", 1),
                    fail_fast=_env(source, "RAG_INGESTION_FAIL_FAST", True),
                ),
                docling=DoclingSettings(
                    enabled=_env(source, "DOCLING_ENABLED", False),
                    strict_mode=_env(source, "DOCLING_STRICT_MODE", False),
                    parser_mode=_env(source, "DOCLING_PARSER_MODE", "auto"),
                    base_url=_env(
                        source,
                        "DOCLING_BASE_URL",
                        "http://127.0.0.1:5001",
                    ),
                    api_key=_env(source, "DOCLING_API_KEY", ""),
                    request_timeout=_env(source, "DOCLING_REQUEST_TIMEOUT", 120.0),
                    task_timeout=_env(source, "DOCLING_TASK_TIMEOUT", 3_600.0),
                    poll_interval=_env(source, "DOCLING_POLL_INTERVAL", 2.0),
                    max_pages_per_task=_env(
                        source,
                        "DOCLING_MAX_PAGES_PER_TASK",
                        3,
                    ),
                    segment_retries=_env(source, "DOCLING_SEGMENT_RETRIES", 1),
                    image_bytes_per_page=_env(
                        source,
                        "DOCLING_AUTO_IMAGE_BYTES_PER_PAGE",
                        307_200,
                    ),
                    min_text_chars_per_page=_env(
                        source,
                        "DOCLING_AUTO_MIN_TEXT_CHARS_PER_PAGE",
                        100,
                    ),
                    min_image_area_ratio=_env(
                        source,
                        "DOCLING_AUTO_MIN_IMAGE_AREA_RATIO",
                        0.15,
                    ),
                    min_table_like_pages_ratio=_env(
                        source,
                        "DOCLING_AUTO_MIN_TABLE_LIKE_PAGES_RATIO",
                        0.6,
                    ),
                    min_multi_column_pages_ratio=_env(
                        source,
                        "DOCLING_AUTO_MIN_MULTI_COLUMN_PAGES_RATIO",
                        0.4,
                    ),
                    sample_pages=_env(source, "DOCLING_AUTO_SAMPLE_PAGES", 5),
                    max_pdf_pages=max_pdf_pages,
                    local_min_page_coverage=_env(
                        source,
                        "DOCLING_LOCAL_MIN_PAGE_COVERAGE",
                        0.65,
                    ),
                    local_min_chars_per_page=_env(
                        source,
                        "DOCLING_LOCAL_MIN_CHARS_PER_PAGE",
                        80,
                    ),
                    local_max_replacement_ratio=_env(
                        source,
                        "DOCLING_LOCAL_MAX_REPLACEMENT_RATIO",
                        0.005,
                    ),
                    local_min_traceable_ratio=_env(
                        source,
                        "DOCLING_LOCAL_MIN_TRACEABLE_RATIO",
                        0.95,
                    ),
                ),
                hash_record_path=_env_path(
                    source,
                    "HASH_RECORD_PATH",
                    "./chroma_db/file_hashes.json",
                ),
            ),
            service=KnowledgeServiceSettings(
                api_key=source.get("KNOWLEDGE_SERVICE_API_KEY", "").strip(),
                require_api_key=_env_bool(
                    source,
                    "KNOWLEDGE_SERVICE_REQUIRE_API_KEY",
                ),
                ingestion_root=Path(
                    source.get(
                        "KNOWLEDGE_SERVICE_INGESTION_ROOT",
                        "./FinancialResearchReportData",
                    )
                ).expanduser(),
                warmup_on_start=_env_bool(
                    source,
                    "KNOWLEDGE_SERVICE_WARMUP_ON_START",
                ),
                evaluation_api_enabled=_env_bool(
                    source,
                    "KNOWLEDGE_SERVICE_EVALUATION_API_ENABLED",
                ),
            ),
        )


def load_client_settings(
    environ: Mapping[str, str] | None = None,
) -> KnowledgeClientSettings:
    source = os.environ if environ is None else environ
    mode = source.get("RAG_RUNTIME_MODE", "remote").strip().lower()
    if mode not in {"local", "remote"}:
        raise ValueError("RAG_RUNTIME_MODE 仅支持 remote 或 local")
    if mode == "local":
        raise RuntimeError(
            "RAG_RUNTIME_MODE=local 只允许 Knowledge Service 或显式离线任务使用；"
            "Agent、MCP 与评测入口必须通过 remote 模式访问知识库"
        )
    base_url = source.get("KNOWLEDGE_SERVICE_URL", "").strip()
    if not base_url:
        raise RuntimeError(
            "RAG_RUNTIME_MODE=remote 时必须配置 KNOWLEDGE_SERVICE_URL；"
            "只有 Knowledge Service 或显式离线任务可以使用 local 模式"
        )
    try:
        timeout = float(source.get("KNOWLEDGE_SERVICE_TIMEOUT", "150"))
    except ValueError:
        timeout = 150.0
    return KnowledgeClientSettings(
        base_url=base_url,
        api_key=source.get("KNOWLEDGE_SERVICE_API_KEY", ""),
        timeout=max(1.0, timeout),
    )


def _env(source: Mapping[str, str], name: str, default: object) -> object:
    return source[name] if name in source else default


def _optional_env(source: Mapping[str, str], name: str) -> object | None:
    return source[name] if name in source else None


def _env_bool(
    source: Mapping[str, str],
    name: str,
    default: bool = False,
) -> bool:
    raw = source.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是布尔值：true/false、1/0、yes/no 或 on/off")


def _env_path(
    source: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    raw = source[name] if name in source else default
    if not raw.strip():
        raise ValueError(f"{name} 不能为空")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path.resolve())


__all__ = [
    "DoclingSettings",
    "IngestionBatchSettings",
    "IngestionSettings",
    "KnowledgeClientSettings",
    "KnowledgeServiceSettings",
    "KnowledgeSettings",
    "RagProfile",
    "RagSettings",
    "RagTuningSettings",
    "RedisSettings",
    "RerankerSettings",
    "SharedSettings",
    "StorageSettings",
    "load_client_settings",
]
