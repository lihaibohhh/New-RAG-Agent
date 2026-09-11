"""LangChain BM25 与内部 RagDocument 之间的适配器。"""
from __future__ import annotations

from typing import Any

from react_agent.rag.contracts import RagDocument


class Bm25CandidateRetriever:
    def __init__(self, backend: Any, *, top_k: int) -> None:
        self.backend = backend
        self.backend.k = top_k

    @classmethod
    def build(
        cls,
        documents: list[RagDocument],
        *,
        top_k: int,
    ) -> "Bm25CandidateRetriever":
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document

        backend = BM25Retriever.from_documents(
            [
                Document(
                    page_content=document.content,
                    metadata=document.metadata.to_dict(),
                    id=document.document_id,
                )
                for document in documents
            ]
        )
        return cls(backend, top_k=top_k)

    def retrieve(self, query: str) -> list[RagDocument]:
        return [
            RagDocument(
                content=str(document.page_content or ""),
                metadata=dict(document.metadata or {}),
                document_id=(
                    str(document.id)
                    if getattr(document, "id", None) not in (None, "")
                    else None
                ),
            )
            for document in (self.backend.invoke(query) or [])
        ]


__all__ = ["Bm25CandidateRetriever"]
