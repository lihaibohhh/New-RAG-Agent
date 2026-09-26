"""顶层组合根持有的共享配置与模块配置聚合。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from knowledge.ingestion.config import IngestionRuntimeConfig
from knowledge.rag.config import RagRuntimeConfig


class _FrozenRuntimeModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class StorageRuntimeConfig(_FrozenRuntimeModel):
    chroma_db_path: str = Field(default="./chroma_db", min_length=1)
    chunk_store_path: str = Field(
        default="./data/knowledge/chunks.sqlite3",
        min_length=1,
    )


class SharedResourceConfig(_FrozenRuntimeModel):
    """跨模块共享模型资源的最小配置。"""

    requested_device: Literal["auto", "cpu", "cuda"] = "auto"
    embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", min_length=1)

    @field_validator("requested_device", mode="before")
    @classmethod
    def _normalize_device(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class KnowledgeRuntimeConfig(_FrozenRuntimeModel):
    """仅由 Server 组合根持有的模块配置聚合。"""

    shared: SharedResourceConfig = Field(default_factory=SharedResourceConfig)
    storage: StorageRuntimeConfig = Field(default_factory=StorageRuntimeConfig)
    rag: RagRuntimeConfig = Field(default_factory=RagRuntimeConfig)
    ingestion: IngestionRuntimeConfig = Field(default_factory=IngestionRuntimeConfig)


__all__ = [
    "KnowledgeRuntimeConfig",
    "SharedResourceConfig",
    "StorageRuntimeConfig",
]
