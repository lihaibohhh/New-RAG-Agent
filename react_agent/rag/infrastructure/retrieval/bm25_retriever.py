"""LangChain BM25 与内部 RagDocument 之间的适配器。"""
from __future__ import annotations

from typing import Any

from react_agent.rag.contracts import RagDocument
from react_agent.rag.infrastructure.retrieval.bm25_tokenizer import tokenize_bm25


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
            ],
            preprocess_func=tokenize_bm25,
        )
        return cls(backend, top_k=top_k)

    def retrieve(self, query: str) -> list[RagDocument]:
        processed_query = self.backend.preprocess_func(query)
        scores = list(self.backend.vectorizer.get_scores(processed_query))
        if not scores or max(scores) <= 0:
            return []
        ranked_indices = sorted(
            range(len(scores)),
            key=scores.__getitem__,
            reverse=True,
        )
        documents = [
            self.backend.docs[index]
            for index in ranked_indices[: self.backend.k]
            if scores[index] > 0
        ]
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
            for document in documents
        ]


__all__ = ["Bm25CandidateRetriever"]
