"""Chroma 向量批量写入适配器。"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from knowledge.contracts import RagDocument
from knowledge.observability import timer


logger = logging.getLogger(__name__)


class ChromaVectorWriterAdapter:
    """只负责初始化 Chroma 并幂等写入已解析的文档批次。"""

    def __init__(
        self,
        *,
        chroma_dir: str,
        embedding_provider: Callable[[], Any],
    ) -> None:
        self._chroma_dir = chroma_dir
        self._embedding_provider = embedding_provider
        self._embeddings: Any | None = None
        self._vectorstore: Any | None = None

    def initialize(self) -> None:
        if self._vectorstore is not None:
            return

        from langchain_chroma import Chroma

        logger.info("[RAG] 加载 Runtime 注入的 embedding 模型")
        with timer("T3_model_load"):
            self._embeddings = self._embedding_provider()
            self._vectorstore = Chroma(
                persist_directory=self._chroma_dir,
                embedding_function=self._embeddings,
            )

    def write(
        self,
        documents: list[RagDocument],
        *,
        batch_label: str,
        global_offset: int,
    ) -> None:
        if not documents:
            return
        self.initialize()
        assert self._embeddings is not None
        assert self._vectorstore is not None

        batch_meta = {"batch_label": batch_label, "chunk_count": len(documents)}
        texts = [document.content for document in documents]
        with timer("T3_embedding", meta=batch_meta):
            vectors = self._embeddings.embed_documents(texts)
        with timer("T4_chroma_write", meta=batch_meta):
            self._vectorstore._collection.upsert(
                ids=[
                    document.metadata.chunk_id
                    or document.document_id
                    or f"chunk_{global_offset + index}"
                    for index, document in enumerate(documents)
                ],
                embeddings=vectors,
                documents=texts,
                metadatas=[
                    document.metadata.to_storage_dict() for document in documents
                ],
            )


__all__ = ["ChromaVectorWriterAdapter"]
