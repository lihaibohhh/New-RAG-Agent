"""RAG 模块独占的运行配置契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _FrozenRagConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class RagProfile(_FrozenRagConfig):
    device: Literal["cpu", "cuda"]
    rerank_candidates: int = Field(ge=1, le=100)
    rerank_top_n: int = Field(ge=1, le=20)
    reranker_concurrency: int = Field(ge=1, le=8)
    reranker_batch_size: int = Field(ge=1, le=128)


class RagTuningConfig(_FrozenRagConfig):
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
        """将设备相关覆盖项收敛为单一运行档位。"""
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


class RedisRuntimeConfig(_FrozenRagConfig):
    url: str = Field(default="redis://localhost:6379", min_length=1)
    max_connections: int = Field(default=20, gt=0)
    semantic_cache_ttl: int = Field(default=3_600, gt=0)
    semantic_cache_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    semantic_cache_min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_index_ttl: int = Field(default=86_400, gt=0)
    bm25_hmac_secret: str = ""


class RerankerRuntimeConfig(_FrozenRagConfig):
    model: str = Field(default="BAAI/bge-reranker-v2-m3", min_length=1)
    threshold: float = Field(default=0.1, ge=0.0, le=1.0, allow_inf_nan=False)
    debug: bool = False


class RagRuntimeConfig(_FrozenRagConfig):
    """仅包含在线检索对象图可见的配置。"""

    tuning: RagTuningConfig = Field(default_factory=RagTuningConfig)
    redis: RedisRuntimeConfig = Field(default_factory=RedisRuntimeConfig)
    reranker: RerankerRuntimeConfig = Field(default_factory=RerankerRuntimeConfig)


__all__ = [
    "RagProfile",
    "RagRuntimeConfig",
    "RagTuningConfig",
    "RedisRuntimeConfig",
    "RerankerRuntimeConfig",
]
