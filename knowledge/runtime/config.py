"""本地知识库 Runtime 的显式配置契约。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from knowledge.ingestion import IngestionConfig


class _FrozenRuntimeModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class RagProfile(_FrozenRuntimeModel):
    device: Literal["cpu", "cuda"]
    rerank_candidates: int = Field(ge=1, le=100)
    rerank_top_n: int = Field(ge=1, le=20)
    reranker_concurrency: int = Field(ge=1, le=8)
    reranker_batch_size: int = Field(ge=1, le=128)


class RagTuningConfig(_FrozenRuntimeModel):
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
            or (
                self.cuda_rerank_candidates
                if is_cuda
                else self.cpu_rerank_candidates
            ),
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


class RedisRuntimeConfig(_FrozenRuntimeModel):
    url: str = Field(default="redis://localhost:6379", min_length=1)
    max_connections: int = Field(default=20, gt=0)
    semantic_cache_ttl: int = Field(default=3_600, gt=0)
    semantic_cache_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    semantic_cache_min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_index_ttl: int = Field(default=86_400, gt=0)
    bm25_hmac_secret: str = ""


class StorageRuntimeConfig(_FrozenRuntimeModel):
    data_dir: str = Field(default="./FinancialResearchReportData", min_length=1)
    chroma_db_path: str = Field(default="./chroma_db", min_length=1)
    hash_record_path: str = Field(
        default="./chroma_db/file_hashes.json",
        min_length=1,
    )
    chunk_store_path: str = Field(
        default="./data/knowledge/chunks.sqlite3",
        min_length=1,
    )


class DoclingRuntimeConfig(_FrozenRuntimeModel):
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


class ModelRuntimeConfig(_FrozenRuntimeModel):
    embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", min_length=1)
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", min_length=1)
    reranker_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    reranker_debug: bool = False


class KnowledgeRuntimeConfig(_FrozenRuntimeModel):
    requested_device: Literal["auto", "cpu", "cuda"] = "auto"
    rag: RagTuningConfig = Field(default_factory=RagTuningConfig)
    redis: RedisRuntimeConfig = Field(default_factory=RedisRuntimeConfig)
    storage: StorageRuntimeConfig = Field(default_factory=StorageRuntimeConfig)
    docling: DoclingRuntimeConfig = Field(default_factory=DoclingRuntimeConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    models: ModelRuntimeConfig = Field(default_factory=ModelRuntimeConfig)

    @field_validator("requested_device", mode="before")
    @classmethod
    def _normalize_device(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


__all__ = [
    "DoclingRuntimeConfig",
    "KnowledgeRuntimeConfig",
    "ModelRuntimeConfig",
    "RagProfile",
    "RagTuningConfig",
    "RedisRuntimeConfig",
    "StorageRuntimeConfig",
]
