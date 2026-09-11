"""RAG 在线查询的唯一业务编排入口。"""
from __future__ import annotations

import logging
import time
from typing import Any

from react_agent.rag.contracts import (
    RagValidationError,
    RetrievalRequest,
    RetrievalResult,
    WarmupStatus,
)
from react_agent.rag.ports import (
    HybridRetrieverPort,
    RerankerPort,
    SemanticCachePort,
)
from react_agent.rag.query.result_builder import build_chunks


logger = logging.getLogger(__name__)


class RetrievalService:
    """编排缓存、混合召回、精排和结果归一化。"""

    def __init__(
        self,
        *,
        cache: SemanticCachePort,
        retriever: HybridRetrieverPort,
        reranker: RerankerPort,
        max_content_chars: int = 1200,
        rerank_candidates: int = 12,
        rerank_top_n: int = 5,
    ) -> None:
        self._cache = cache
        self._retriever = retriever
        self._reranker = reranker
        self._max_content_chars = max(200, min(int(max_content_chars), 5000))
        self._rerank_candidates = max(1, min(int(rerank_candidates), 100))
        self._rerank_top_n = max(1, min(int(rerank_top_n), 20))

    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        use_query_cache: bool = True,
        retrieval_mode: str = "hybrid",
    ) -> RetrievalResult:
        """执行检索；评测可关闭查询结果缓存以避免跨实验污染。"""
        total_started = time.perf_counter()
        timings: dict[str, float] = {}
        request = RetrievalRequest(query=query, top_k=top_k, filters=filters).normalized()
        mode = str(retrieval_mode or "hybrid").strip().lower()
        if mode not in {"hybrid", "bm25", "vector"}:
            raise RagValidationError(f"不支持的检索模式: {retrieval_mode}")
        query_cache_enabled = (
            use_query_cache and request.filters is None and mode == "hybrid"
        )

        # 当前 Redis 语义缓存以 query 为键，不包含 filters；带过滤条件时跳过缓存，
        # 防止复用未过滤结果。基础设施召回器会对候选元数据做精确过滤。
        cached_documents = None
        stage_started = time.perf_counter()
        if query_cache_enabled:
            cached_documents = await self._cache.get(request.query)
        timings["cache_get"] = round(time.perf_counter() - stage_started, 4)
        if cached_documents is not None:
            chunks = build_chunks(
                list(cached_documents)[:request.top_k],
                max_chars=self._max_content_chars,
            )
            return RetrievalResult(
                query=request.query,
                chunks=chunks,
                stage="semantic_cache_hit",
                cache_hit=True,
                timings={
                    **timings,
                    "total": round(time.perf_counter() - total_started, 4),
                },
            )

        stage_started = time.perf_counter()
        if mode == "hybrid":
            candidates = await self._retriever.retrieve(
                request.query,
                filters=request.filters,
            )
        else:
            candidates = await self._retriever.retrieve(
                request.query,
                filters=request.filters,
                mode=mode,
            )
        timings["retrieve"] = round(time.perf_counter() - stage_started, 4)
        if not candidates:
            if query_cache_enabled:
                await self._cache.set(request.query, [], top_score=0.0)
            return RetrievalResult(
                query=request.query,
                stage=("dual_retrieve_empty" if mode == "hybrid" else f"{mode}_retrieve_empty"),
                timings={
                    **timings,
                    "total": round(time.perf_counter() - total_started, 4),
                },
            )

        # 缓存候选池允许后续更大的 top_k 请求复用，避免首次 top_k=1
        # 导致同一 query 的缓存永久只剩一条。
        rerank_input = candidates[: self._rerank_candidates]
        stage_started = time.perf_counter()
        documents, top_score = await self._reranker.rerank(
            request.query,
            rerank_input,
            top_n=max(request.top_k, self._rerank_top_n),
        )
        timings["rerank"] = round(time.perf_counter() - stage_started, 4)
        if query_cache_enabled:
            stage_started = time.perf_counter()
            await self._cache.set(
                request.query,
                documents,
                top_score=float(top_score or 0.0),
            )
            timings["cache_set"] = round(time.perf_counter() - stage_started, 4)
        chunks = build_chunks(
            documents[:request.top_k],
            max_chars=self._max_content_chars,
        )
        timings["total"] = round(time.perf_counter() - total_started, 4)
        stage = "dual_retrieve+rerank" if mode == "hybrid" else f"{mode}_retrieve+rerank"
        logger.info(
            "[RAG] query complete stage=%s candidates=%s "
            "reranked=%s returned=%s timings=%s",
            stage,
            len(candidates),
            len(rerank_input),
            len(chunks),
            timings,
        )
        return RetrievalResult(
            query=request.query,
            chunks=chunks,
            stage=stage,
            candidates_count=len(candidates),
            reranked_count=len(rerank_input),
            top_score=float(top_score or 0.0),
            timings=timings,
        )

    async def warmup(self) -> WarmupStatus:
        timings: dict[str, float] = {}

        started = time.perf_counter()
        await self._retriever.warmup()
        timings["warm_retriever"] = round(time.perf_counter() - started, 4)

        started = time.perf_counter()
        await self._reranker.warmup()
        timings["warm_reranker"] = round(time.perf_counter() - started, 4)
        timings["total"] = round(sum(timings.values()), 4)
        return WarmupStatus(ready=True, timings=timings)
