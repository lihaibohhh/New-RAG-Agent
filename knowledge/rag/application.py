"""RAG 查询主流程：缓存、双路召回、融合、过滤与精排。"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Iterable, Sequence
from typing import Any

from knowledge.contracts import (
    EvaluationCandidate,
    EvaluationRetrievalResult,
    KnowledgeDocument,
    KnowledgeValidationError,
    RetrievalResult,
    RetrievedChunk,
    WarmupStatus,
)
from knowledge.rag.contracts import (
    Bm25CandidateRetrieverPort,
    CandidateRetrievalTrace,
    RerankerPort,
    RetrievalRequest,
    SemanticCachePort,
    VectorCandidateRetrieverPort,
)

logger = logging.getLogger(__name__)


class RetrievalApplicationService:
    """项目唯一的查询编排：缓存 -> 混合召回 -> RRF -> 精排。"""

    def __init__(
        self,
        *,
        cache: SemanticCachePort,
        vector_retriever: VectorCandidateRetrieverPort,
        bm25_retriever: Bm25CandidateRetrieverPort,
        reranker: RerankerPort,
        max_content_chars: int = 1200,
        source_top_k: int = 10,
        rrf_rank_constant: int = 60,
        rerank_candidates: int = 12,
        rerank_top_n: int = 5,
        reranker_batch_size: int = 1,
        device: str = "unknown",
    ) -> None:
        self._cache = cache
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._reranker = reranker
        self._max_content_chars = max(200, min(int(max_content_chars), 5000))
        self._source_top_k = max(1, int(source_top_k))
        self._rrf_rank_constant = max(1, int(rrf_rank_constant))
        self._rerank_candidates = max(1, min(int(rerank_candidates), 100))
        self._rerank_top_n = max(1, min(int(rerank_top_n), 20))
        self._reranker_batch_size = max(1, int(reranker_batch_size))
        self._device = str(device or "unknown")

    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        use_query_cache: bool = True,
        retrieval_mode: str = "hybrid",
    ) -> RetrievalResult:
        total_started = time.perf_counter()
        request = RetrievalRequest(query=query, top_k=top_k, filters=filters).normalized()
        mode = self._normalize_mode(retrieval_mode)
        timings: dict[str, float] = {}
        cache_enabled = use_query_cache and request.filters is None and mode == "hybrid"

        stage_started = time.perf_counter()
        cached_documents = await self._cache.get(request.query) if cache_enabled else None
        timings["cache_get"] = round(time.perf_counter() - stage_started, 4)
        if cached_documents is not None:
            return RetrievalResult(
                query=request.query,
                chunks=build_chunks(
                    list(cached_documents)[: request.top_k],
                    max_chars=self._max_content_chars,
                ),
                stage="semantic_cache_hit",
                cache_hit=True,
                timings={**timings, "total": round(time.perf_counter() - total_started, 4)},
            )

        trace = await self._retrieve_with_trace(
            request.query, filters=request.filters, mode=mode
        )
        timings.update(trace.timings)
        candidates = list(trace.filtered_candidates)
        if not candidates:
            if cache_enabled:
                await self._cache.set(request.query, [], top_score=0.0)
            return RetrievalResult(
                query=request.query,
                stage="dual_retrieve_empty" if mode == "hybrid" else f"{mode}_retrieve_empty",
                timings={**timings, "total": round(time.perf_counter() - total_started, 4)},
            )

        rerank_input = candidates[: self._rerank_candidates]
        stage_started = time.perf_counter()
        documents, top_score = await self._reranker.rerank(
            request.query,
            rerank_input,
            top_n=max(request.top_k, self._rerank_top_n),
        )
        timings["rerank"] = round(time.perf_counter() - stage_started, 4)
        if cache_enabled:
            stage_started = time.perf_counter()
            await self._cache.set(
                request.query, documents, top_score=float(top_score or 0.0)
            )
            timings["cache_set"] = round(time.perf_counter() - stage_started, 4)

        chunks = build_chunks(documents[: request.top_k], max_chars=self._max_content_chars)
        timings["total"] = round(time.perf_counter() - total_started, 4)
        stage = "dual_retrieve+rerank" if mode == "hybrid" else f"{mode}_retrieve+rerank"
        logger.info(
            "[RAG] query complete stage=%s candidates=%s returned=%s timings=%s",
            stage,
            len(candidates),
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

    async def evaluation_search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        retrieval_mode: str = "hybrid",
        use_query_cache: bool = False,
    ) -> EvaluationRetrievalResult:
        """返回各阶段快照；单路召回模式只保留给评测对比。"""
        if use_query_cache:
            raise KnowledgeValidationError("评测检索管道不允许使用语义查询缓存")
        total_started = time.perf_counter()
        request = RetrievalRequest(query=query, top_k=top_k, filters=filters).normalized()
        mode = self._normalize_mode(retrieval_mode)
        trace = await self._retrieve_with_trace(
            request.query, filters=request.filters, mode=mode
        )
        candidates = list(trace.filtered_candidates)
        rerank_input = candidates[: self._rerank_candidates]
        rerank_started = time.perf_counter()
        if rerank_input:
            reranked, _ = await self._reranker.rerank(
                request.query, rerank_input, top_n=len(rerank_input)
            )
        else:
            reranked = []
        final_documents = reranked[: request.top_k]
        stages = {
            "bm25": self._snapshots(trace.bm25_candidates),
            "vector": self._snapshots(trace.vector_candidates),
            "fusion": self._snapshots(trace.fusion_candidates),
            "filtered": self._snapshots(trace.filtered_candidates),
            "reranker_input": self._snapshots(rerank_input),
            "reranked": self._snapshots(reranked),
            "final": self._snapshots(final_documents),
        }
        timings = {
            **trace.timings,
            "reranker": round(time.perf_counter() - rerank_started, 4),
            "evaluation_total": round(time.perf_counter() - total_started, 4),
        }
        return EvaluationRetrievalResult(
            query=request.query,
            retrieval_mode=mode,
            chunks=build_chunks(final_documents, max_chars=self._max_content_chars),
            stages=stages,
            timings=timings,
            configuration={
                "query_cache_enabled": False,
                "device": self._device,
                "source_top_k": self._source_top_k,
                "fusion_limit": 2 * self._source_top_k,
                "rrf_rank_constant": self._rrf_rank_constant,
                "rerank_candidates": self._rerank_candidates,
                "production_rerank_top_n": self._rerank_top_n,
                "reranker_batch_size": self._reranker_batch_size,
                "final_top_k": request.top_k,
            },
            degraded_sources=trace.degraded_sources,
        )

    async def _retrieve_with_trace(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        mode: str,
    ) -> CandidateRetrievalTrace:
        total_started = time.perf_counter()
        timings: dict[str, float] = {}
        degraded_sources: list[str] = []

        if mode == "vector":
            started = time.perf_counter()
            documents = await asyncio.to_thread(self._vector.retrieve, query)
            timings["vector"] = round(time.perf_counter() - started, 4)
            filtered = self._timed_filter(documents, filters, timings)
            timings["total"] = round(time.perf_counter() - total_started, 4)
            return CandidateRetrievalTrace(
                retrieval_mode=mode,
                vector_candidates=tuple(documents),
                fusion_candidates=tuple(documents),
                filtered_candidates=tuple(filtered),
                timings=timings,
            )

        started = time.perf_counter()
        bm25_available = await asyncio.to_thread(self._bm25.prepare)
        timings["bm25_prepare"] = round(time.perf_counter() - started, 4)
        if mode == "bm25":
            started = time.perf_counter()
            documents = (
                await asyncio.to_thread(self._bm25.retrieve, query)
                if bm25_available
                else []
            )
            timings["bm25"] = round(time.perf_counter() - started, 4)
            filtered = self._timed_filter(documents, filters, timings)
            timings["total"] = round(time.perf_counter() - total_started, 4)
            return CandidateRetrievalTrace(
                retrieval_mode=mode,
                bm25_candidates=tuple(documents),
                fusion_candidates=tuple(documents),
                filtered_candidates=tuple(filtered),
                timings=timings,
            )

        async def retrieve_source(name: str, call):
            source_started = time.perf_counter()
            try:
                return await asyncio.to_thread(call), None
            except Exception as exc:
                return [], exc
            finally:
                timings[name] = round(time.perf_counter() - source_started, 4)

        if not bm25_available:
            bm25_documents: list[KnowledgeDocument] = []
            vector_documents, vector_error = await retrieve_source(
                "vector", lambda: self._vector.retrieve(query)
            )
            if vector_error is not None:
                raise vector_error
            degraded_sources.append("bm25_unavailable")
        else:
            (bm25_documents, bm25_error), (vector_documents, vector_error) = (
                await asyncio.gather(
                    retrieve_source("bm25", lambda: self._bm25.retrieve(query)),
                    retrieve_source("vector", lambda: self._vector.retrieve(query)),
                )
            )
            if bm25_error is not None and vector_error is not None:
                raise RuntimeError(
                    "BM25 与向量召回同时失败: "
                    f"bm25={bm25_error}; vector={vector_error}"
                ) from vector_error
            if bm25_error is not None:
                logger.warning("[RAG] BM25 召回失败，降级为纯向量召回: %s", bm25_error)
                degraded_sources.append("bm25_error")
            if vector_error is not None:
                logger.warning("[RAG] 向量召回失败，降级为纯 BM25 召回: %s", vector_error)
                degraded_sources.append("vector_error")

        started = time.perf_counter()
        documents = _reciprocal_rank_fusion(
            (bm25_documents, vector_documents),
            rank_constant=self._rrf_rank_constant,
            limit=2 * self._source_top_k,
        )
        timings["fusion"] = round(time.perf_counter() - started, 4)
        filtered = self._timed_filter(documents, filters, timings)
        timings["total"] = round(time.perf_counter() - total_started, 4)
        return CandidateRetrievalTrace(
            retrieval_mode=mode,
            bm25_candidates=tuple(bm25_documents),
            vector_candidates=tuple(vector_documents),
            fusion_candidates=tuple(documents),
            filtered_candidates=tuple(filtered),
            degraded_sources=tuple(degraded_sources),
            timings=timings,
        )

    async def warmup(self) -> WarmupStatus:
        started = time.perf_counter()
        bm25_result, vector_result = await asyncio.gather(
            asyncio.to_thread(self._bm25.prepare),
            asyncio.to_thread(self._vector.warmup),
            return_exceptions=True,
        )
        if isinstance(bm25_result, BaseException) and isinstance(
            vector_result, BaseException
        ):
            raise RuntimeError(
                "BM25 与向量召回预热同时失败: "
                f"bm25={bm25_result}; vector={vector_result}"
            ) from vector_result
        if isinstance(bm25_result, BaseException):
            logger.warning("[RAG] BM25 预热失败，保留纯向量能力: %s", bm25_result)
        if isinstance(vector_result, BaseException):
            logger.warning("[RAG] 向量预热失败，保留纯 BM25 能力: %s", vector_result)
        retrieval_elapsed = round(time.perf_counter() - started, 4)
        started = time.perf_counter()
        await self._reranker.warmup()
        reranker_elapsed = round(time.perf_counter() - started, 4)
        return WarmupStatus(
            ready=True,
            timings={
                "warm_retriever": retrieval_elapsed,
                "warm_reranker": reranker_elapsed,
                "total": round(retrieval_elapsed + reranker_elapsed, 4),
            },
        )

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode or "hybrid").strip().lower()
        if normalized not in {"hybrid", "bm25", "vector"}:
            raise KnowledgeValidationError(f"不支持的检索模式: {mode}")
        return normalized

    @staticmethod
    def _timed_filter(
        documents: list[KnowledgeDocument],
        filters: dict[str, Any] | None,
        timings: dict[str, float],
    ) -> list[KnowledgeDocument]:
        started = time.perf_counter()
        filtered = (
            documents
            if not filters
            else [item for item in documents if item.metadata.matches(filters)]
        )
        timings["filter"] = round(time.perf_counter() - started, 4)
        return filtered

    @staticmethod
    def _snapshots(
        documents: Sequence[KnowledgeDocument],
    ) -> tuple[EvaluationCandidate, ...]:
        return tuple(
            EvaluationCandidate(
                rank=rank,
                chunk_id=document.metadata.chunk_id,
                source_file=document.metadata.resolved_source_file(),
                source_page=document.metadata.source_page,
                content_chars=len(document.content or ""),
                doc_type=document.metadata.doc_type,
                industry=document.metadata.industry,
            )
            for rank, document in enumerate(documents, 1)
        )


def _reciprocal_rank_fusion(
    rankings: Iterable[Sequence[KnowledgeDocument]],
    *,
    rank_constant: int,
    limit: int,
) -> list[KnowledgeDocument]:
    scores: dict[str, float] = {}
    documents: dict[str, KnowledgeDocument] = {}
    for ranking in rankings:
        for rank, document in enumerate(ranking):
            key = str(document.metadata.chunk_id or document.document_id or "")
            if not key:
                key = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            scores[key] = scores.get(key, 0.0) + 1 / (rank_constant + rank + 1)
            documents[key] = document
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [documents[key] for key in ordered[:limit]]


def build_chunks(
    documents: list[KnowledgeDocument],
    *,
    max_chars: int,
) -> tuple[RetrievedChunk, ...]:
    """把内部候选归一化为公共检索结果。"""
    chunks: list[RetrievedChunk] = []
    for document in documents:
        content = str(document.content or "").strip()
        if not content:
            continue
        metadata = document.metadata
        chunks.append(
            RetrievedChunk(
                content=_trim_text(content, max_chars),
                source_file=metadata.resolved_source_file(),
                source_page=metadata.source_page,
                chunk_id=metadata.chunk_id,
                score=metadata.retrieval_score,
                doc_type=metadata.doc_type,
                industry=metadata.industry,
            )
        )
    return tuple(chunks)


def _trim_text(text: str, max_chars: int) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


__all__ = ["RetrievalApplicationService", "build_chunks"]
