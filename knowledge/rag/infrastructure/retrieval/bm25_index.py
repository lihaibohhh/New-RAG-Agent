"""BM25 索引的懒加载、持久化和失效适配器。"""
from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from knowledge.contracts import KnowledgeDocument
from knowledge.rag.infrastructure.retrieval.bm25_retriever import (
    Bm25CandidateRetriever,
)
from knowledge.rag.ports import Bm25RepositoryPort, ChunkCorpusPort


logger = logging.getLogger(__name__)


class LazyBm25CandidateRetrieverAdapter:
    """封装 BM25 的构建、Redis 快照、代次失效与后台保存。"""

    def __init__(
        self,
        *,
        repository: Bm25RepositoryPort,
        corpus_reader: ChunkCorpusPort,
        top_k: int = 10,
    ) -> None:
        self._repository = repository
        self._corpus = corpus_reader
        self._top_k = max(1, int(top_k))
        self._retriever: Bm25CandidateRetriever | None = None
        self._lock = threading.Lock()
        self._generation = 0
        self._save_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bm25-redis-save",
        )

    def prepare(self) -> bool:
        """确保 BM25 可用；空语料返回 False 而不是伪造空索引。"""
        return self._get_retriever() is not None

    def retrieve(self, query: str) -> list[KnowledgeDocument]:
        retriever = self._get_retriever()
        return retriever.retrieve(query) if retriever is not None else []

    def invalidate(self) -> None:
        with self._lock:
            self._generation += 1
            self._repository.clear()
            self._retriever = None
        logger.info("[RAG] BM25 Redis 与内存索引已失效")

    async def close(self) -> None:
        await asyncio.to_thread(self._save_executor.shutdown, wait=True)

    def _get_retriever(self) -> Bm25CandidateRetriever | None:
        if self._retriever is not None:
            return self._retriever
        with self._lock:
            if self._retriever is not None:
                return self._retriever

            backend = self._repository.load()
            if backend is not None:
                self._retriever = Bm25CandidateRetriever(
                    backend,
                    top_k=self._top_k,
                )
                return self._retriever

            documents = self._corpus.list_documents()
            if not documents:
                logger.info("[RAG] 知识库为空，本次不创建 BM25 索引")
                return None

            self._retriever = Bm25CandidateRetriever.build(
                documents,
                top_k=self._top_k,
            )
            generation = self._generation
            self._save_executor.submit(
                self._save_if_current,
                self._retriever.backend,
                len(documents),
                generation,
            )
            logger.info("[RAG] BM25 索引重建完成 documents=%s", len(documents))
            return self._retriever

    def _save_if_current(
        self,
        backend: object,
        document_count: int,
        generation: int,
    ) -> None:
        with self._lock:
            if generation != self._generation:
                logger.info("[RAG] 跳过已失效代次的 BM25 Redis 写入")
                return
            self._repository.save(backend, document_count)


__all__ = ["LazyBm25CandidateRetrieverAdapter"]
