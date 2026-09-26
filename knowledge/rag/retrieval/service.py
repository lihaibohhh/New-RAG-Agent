"""BM25、向量召回、融合、过滤和降级的 RAG 策略服务。"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time

from knowledge.contracts import KnowledgeDocument
from knowledge.rag.contracts import CandidateRetrievalTrace
from knowledge.rag.ports import (
    Bm25CandidateRetrieverPort,
    VectorCandidateRetrieverPort,
)
from knowledge.rag.retrieval.rrf import reciprocal_rank_fusion


logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """组合候选源，不拥有存储、模型或索引持久化实现。"""

    def __init__(
        self,
        *,
        vector_retriever: VectorCandidateRetrieverPort,
        bm25_retriever: Bm25CandidateRetrieverPort,
        top_k_per_source: int = 10,
        rrf_rank_constant: int = 60,
    ) -> None:
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._top_k = max(1, int(top_k_per_source))
        self._rrf_rank_constant = max(1, int(rrf_rank_constant))

    async def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, object] | None = None,
        mode: str = "hybrid",
    ) -> list[KnowledgeDocument]:
        trace = await self.retrieve_with_trace(
            query,
            filters=filters,
            mode=mode,
        )
        return list(trace.filtered_candidates)

    async def retrieve_with_trace(
        self,
        query: str,
        *,
        filters: dict[str, object] | None = None,
        mode: str = "hybrid",
    ) -> CandidateRetrievalTrace:
        total_started = time.perf_counter()
        timings: dict[str, float] = {}
        degraded_sources: list[str] = []
        if mode not in {"hybrid", "bm25", "vector"}:
            raise ValueError(f"不支持的检索模式: {mode}")

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
                "vector",
                lambda: self._vector.retrieve(query),
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
        documents = reciprocal_rank_fusion(
            (bm25_documents, vector_documents),
            rank_constant=self._rrf_rank_constant,
            limit=2 * self._top_k,
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

    @classmethod
    def _timed_filter(
        cls,
        documents: list[KnowledgeDocument],
        filters: dict[str, object] | None,
        timings: dict[str, float],
    ) -> list[KnowledgeDocument]:
        started = time.perf_counter()
        filtered = cls._apply_filters(documents, filters)
        timings["filter"] = round(time.perf_counter() - started, 4)
        return filtered

    @staticmethod
    def _apply_filters(
        documents: list[KnowledgeDocument],
        filters: dict[str, object] | None,
    ) -> list[KnowledgeDocument]:
        if not filters:
            return documents
        return [
            document
            for document in documents
            if document.metadata.matches(filters)
        ]

    async def warmup(self) -> None:
        bm25_result, vector_result = await asyncio.gather(
            asyncio.to_thread(self._bm25.prepare),
            asyncio.to_thread(self._vector.warmup),
            return_exceptions=True,
        )
        if isinstance(bm25_result, BaseException) and isinstance(
            vector_result,
            BaseException,
        ):
            raise RuntimeError(
                "BM25 与向量召回预热同时失败: "
                f"bm25={bm25_result}; vector={vector_result}"
            ) from vector_result
        if isinstance(bm25_result, BaseException):
            logger.warning("[RAG] BM25 预热失败，保留纯向量能力: %s", bm25_result)
        elif bm25_result is False:
            logger.info("[RAG] BM25 语料为空，保留纯向量能力")
        if isinstance(vector_result, BaseException):
            logger.warning("[RAG] 向量预热失败，保留纯 BM25 能力: %s", vector_result)

    async def invalidate(self) -> None:
        await asyncio.to_thread(self._bm25.invalidate)

    async def close(self) -> None:
        await self._bm25.close()
        close = getattr(self._vector, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


__all__ = ["HybridRetrievalService"]
