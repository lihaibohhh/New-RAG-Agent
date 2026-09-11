"""BM25、Chroma 与 RRF 的混合召回编排适配器。"""
from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from react_agent.rag.contracts import RagDocument
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
        if mode not in {"hybrid", "bm25", "vector"}:
            raise ValueError(f"不支持的检索模式: {mode}")

        if mode == "vector":
            documents = await asyncio.to_thread(self._vector.retrieve, query)
            return self._apply_filters(documents, filters)

        bm25 = await asyncio.to_thread(self._get_bm25)
        if mode == "bm25":
            documents = (
                await asyncio.to_thread(bm25.retrieve, query)
                if bm25 is not None
                else []
            )
            return self._apply_filters(documents, filters)

        vector_task = asyncio.to_thread(self._vector.retrieve, query)
        if bm25 is None:
            bm25_documents: list[RagDocument] = []
            vector_documents = await vector_task
        else:
            bm25_result, vector_result = await asyncio.gather(
                asyncio.to_thread(bm25.retrieve, query),
                vector_task,
                return_exceptions=True,
            )
            if isinstance(bm25_result, BaseException) and isinstance(
                vector_result, BaseException
            ):
                raise RuntimeError(
                    "BM25 与向量召回同时失败: "
                    f"bm25={bm25_result}; vector={vector_result}"
                ) from vector_result
            if isinstance(bm25_result, BaseException):
                logger.warning("[RAG] BM25 召回失败，降级为纯向量召回: %s", bm25_result)
                bm25_documents = []
            else:
                bm25_documents = bm25_result
            if isinstance(vector_result, BaseException):
                logger.warning("[RAG] 向量召回失败，降级为纯 BM25 召回: %s", vector_result)
                vector_documents = []
            else:
                vector_documents = vector_result

        documents = reciprocal_rank_fusion(
            (bm25_documents, vector_documents),
            rank_constant=self._rrf_rank_constant,
            limit=2 * self._top_k,
        )
        return self._apply_filters(documents, filters)

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
