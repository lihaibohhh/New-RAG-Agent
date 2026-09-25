"""Knowledge Service 的版本化 HTTP 请求契约。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    filters: dict[str, Any] | None = None
    use_query_cache: bool = True
    retrieval_mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


class EvaluationSearchRequest(BaseModel):
    """专用评测检索请求；设计上不提供语义查询缓存开关。"""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    filters: dict[str, Any] | None = None
    retrieval_mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


class WarmupRequest(BaseModel):
    wait_seconds: float = Field(default=20, ge=0, le=120)
    force: bool = False


class IngestionRequest(BaseModel):
    """相对于服务端允许目录的路径，禁止客户端传宿主机绝对路径。"""

    relative_path: str = Field(default=".", min_length=1, max_length=1000)


class InvalidateRequest(BaseModel):
    knowledge_base_id: str = Field(default="default", max_length=100)


__all__ = [
    "IngestionRequest",
    "EvaluationSearchRequest",
    "InvalidateRequest",
    "SearchRequest",
    "WarmupRequest",
]
