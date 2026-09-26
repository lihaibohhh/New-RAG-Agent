"""建库模块独占的运行配置契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _FrozenIngestionConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class IngestionConfig(_FrozenIngestionConfig):
    """增量建库服务的批处理策略。"""

    max_pdf_pages: int = Field(default=200, gt=0)
    batch_size: int = Field(default=1_000, ge=1, le=20_000)
    workers: int = Field(default=1, ge=1, le=16)
    fail_fast: bool = True


class DoclingRuntimeConfig(_FrozenIngestionConfig):
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


class IngestionRuntimeConfig(_FrozenIngestionConfig):
    """仅包含文档解析和增量建库对象图可见的配置。"""

    service: IngestionConfig = Field(default_factory=IngestionConfig)
    docling: DoclingRuntimeConfig = Field(default_factory=DoclingRuntimeConfig)
    hash_record_path: str = Field(
        default="./chroma_db/file_hashes.json",
        min_length=1,
    )


__all__ = [
    "DoclingRuntimeConfig",
    "IngestionConfig",
    "IngestionRuntimeConfig",
]
