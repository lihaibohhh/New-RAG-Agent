"""BM25、Chroma 与 RRF 的混合召回编排适配器。"""
from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from react_agent.rag.contracts import CandidateRetrievalTrace, RagDocument
from react_agent.rag.infrastructure.retrieval.bm25_retriever import (
    Bm25CandidateRetriever,
)
from react_agent.rag.infrastructure.retrieval.component_ports import (
    Bm25RepositoryPort,
    ChunkCorpusPort,
    VectorCandidateRetrieverPort,
)
from react_agent.rag.infrastructure.retrieval.rrf import reciprocal_rank_fusion


logger = logging.getLogger(__name__)


class HybridRetrieverAdapter:
    """组合两个候选源；不拥有其存储、序列化或融合算法实现。"""

    def __init__(
        self,
        *,
        vector_retriever: VectorCandidateRetrieverPort,
        bm25_repository: Bm25RepositoryPort,
        corpus_reader: ChunkCorpusPort | None = None,
        top_k_per_source: int = 10,
        rrf_rank_constant: int = 60,
    ) -> None:
        self._vector = vector_retriever
        self._corpus = corpus_reader or vector_retriever
        self._repository = bm25_repository
        self._top_k = top_k_per_source
        self._rrf_rank_constant = rrf_rank_constant
        self._bm25: Bm25CandidateRetriever | None = None
        self._bm25_lock = threading.Lock()
        self._generation = 0
        self._save_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bm25-redis-save",
        )

    async def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, object] | None = None,
        mode: str = "hybrid",
    ) -> list[RagDocument]:
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
        """执行与线上相同的候选召回，并保留评测所需阶段快照。"""
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
        bm25 = await asyncio.to_thread(self._get_bm25)
        timings["bm25_prepare"] = round(time.perf_counter() - started, 4)
        if mode == "bm25":
            started = time.perf_counter()
            documents = (
                await asyncio.to_thread(bm25.retrieve, query)
                if bm25 is not None
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

        if bm25 is None:
            bm25_documents: list[RagDocument] = []
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
                    retrieve_source("bm25", lambda: bm25.retrieve(query)),
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
        documents: list[RagDocument],
        filters: dict[str, object] | None,
        timings: dict[str, float],
    ) -> list[RagDocument]:
        started = time.perf_counter()
        filtered = cls._apply_filters(documents, filters)
        timings["filter"] = round(time.perf_counter() - started, 4)
        return filtered

    @staticmethod
    def _apply_filters(
        documents: list[RagDocument],
        filters: dict[str, object] | None,
    ) -> list[RagDocument]:
        if not filters:
            return documents
        return [
            document
            for document in documents
            if document.metadata.matches(filters)
        ]

    async def warmup(self) -> None:
        prepare = getattr(self._corpus, "prepare", None)
        if prepare is not None:
            try:
                await asyncio.to_thread(prepare)
            except Exception as exc:
                logger.warning("[RAG] Chunk Store 准备失败，将尝试现有召回源: %s", exc)
        bm25_result, vector_result = await asyncio.gather(
            asyncio.to_thread(self._get_bm25),
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

    async def invalidate(self) -> None:
        await asyncio.to_thread(self._invalidate_sync)

    async def close(self) -> None:
        """Wait for pending index persistence and stop the owned executor."""
        await asyncio.to_thread(self._save_executor.shutdown, wait=True)
        close = getattr(self._vector, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    def _get_bm25(self) -> Bm25CandidateRetriever | None:
        if self._bm25 is not None:
            return self._bm25
        with self._bm25_lock:
            if self._bm25 is not None:
                return self._bm25

            backend = self._repository.load()
            if backend is not None:
                self._bm25 = Bm25CandidateRetriever(
                    backend,
                    top_k=self._top_k,
                )
                return self._bm25

            documents = self._corpus.list_documents()
            if not documents:
                logger.info("[RAG] 知识库为空，本次降级为纯向量检索")
                return None
            self._bm25 = Bm25CandidateRetriever.build(
                documents,
                top_k=self._top_k,
            )
            generation = self._generation
            self._save_executor.submit(
                self._save_if_current,
                self._bm25.backend,
                len(documents),
                generation,
            )
            logger.info("[RAG] BM25 索引重建完成 documents=%s", len(documents))
            return self._bm25

    def _invalidate_sync(self) -> None:
        with self._bm25_lock:
            self._generation += 1
            self._repository.clear()
            self._bm25 = None
        logger.info("[RAG] BM25 Redis 与内存索引已失效")

    def _save_if_current(
        self,
        backend: object,
        document_count: int,
        generation: int,
    ) -> None:
        # 与 invalidate 共用锁：旧代保存要么先完成再被 clear，要么被直接跳过。
        with self._bm25_lock:
            if generation != self._generation:
                logger.info("[RAG] 跳过已失效代次的 BM25 Redis 写入")
                return
            self._repository.save(backend, document_count)


__all__ = ["HybridRetrieverAdapter"]
