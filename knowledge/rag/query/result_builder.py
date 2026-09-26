"""把基础设施文档对象归一化为知识库公共契约。"""
from __future__ import annotations

import logging

from knowledge.contracts import KnowledgeDocument, RetrievedChunk


logger = logging.getLogger(__name__)


def build_chunks(
    documents: list[KnowledgeDocument],
    *,
    max_chars: int,
) -> tuple[RetrievedChunk, ...]:
    chunks: list[RetrievedChunk] = []
    for document in documents:
        content = str(document.content or "").strip()
        if not content:
            logger.warning(
                "[RAG] 跳过缺少 content 的检索结果: type=%s",
                type(document).__name__,
            )
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
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
