"""Chroma 向量候选召回适配器。"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from knowledge.contracts import RagDocument
from knowledge.infrastructure.storage.chroma_knowledge_base import (
    validate_chroma_directory,
)
class ChromaCandidateRetriever:
    def __init__(
        self,
        *,
        chroma_dir: str,
        top_k: int,
        embedding_provider: Callable[[], Any],
    ) -> None:
        self._chroma_dir = chroma_dir
        self._top_k = top_k
        self._embedding_provider = embedding_provider
        self._vectorstore: Any | None = None
        self._lock = threading.Lock()

    def _get_vectorstore(self) -> Any:
        if self._vectorstore is not None:
            return self._vectorstore
        with self._lock:
            if self._vectorstore is not None:
                return self._vectorstore
            from langchain_chroma import Chroma

            chroma_dir = str(validate_chroma_directory(self._chroma_dir))
            self._vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=self._embedding_provider(),
            )
            return self._vectorstore

    def retrieve(self, query: str) -> list[RagDocument]:
        documents = self._get_vectorstore().similarity_search(query, k=self._top_k)
        return [_from_langchain(document) for document in documents or []]

    def list_documents(self) -> list[RagDocument]:
        result = self._get_vectorstore().get(include=["documents", "metadatas"])
        if not result or not result.get("documents"):
            return []
        ids = list(result.get("ids") or [])
        texts = list(result.get("documents") or [])
        metadatas = list(result.get("metadatas") or [])
        documents: list[RagDocument] = []
        for index, text in enumerate(texts):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            document_id = str(ids[index]) if index < len(ids) else None
            documents.append(
                RagDocument(
                    content=str(text or ""),
                    metadata=metadata,
                    document_id=document_id,
                )
            )
        return documents

    def warmup(self) -> None:
        self._get_vectorstore()
        self._embedding_provider().embed_query("知识库检索预热")

    async def close(self) -> None:
        """Release the lazily created Chroma wrapper."""
        with self._lock:
            self._vectorstore = None


def _from_langchain(document: Any) -> RagDocument:
    document_id = getattr(document, "id", None)
    return RagDocument(
        content=str(document.page_content or ""),
        metadata=dict(document.metadata or {}),
        document_id=str(document_id) if document_id not in (None, "") else None,
    )


__all__ = ["ChromaCandidateRetriever"]
