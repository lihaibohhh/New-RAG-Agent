"""RAG 离线任务使用的最小 Chroma Chunk 读取入口。"""

from __future__ import annotations

import asyncio

from knowledge.contracts import StoredChunk
from knowledge.rag.infrastructure.storage import ChromaKnowledgeBaseAdapter


async def read_chunks(
    *,
    chroma_dir: str,
    source_file: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[StoredChunk, ...]:
    """直接读取指定 Chroma，不创建完整 Knowledge Runtime。"""
    reader = ChromaKnowledgeBaseAdapter(chroma_dir=chroma_dir)
    chunks = await reader.list_chunks(
        source_file=source_file,
        offset=offset,
        limit=limit,
    )
    return tuple(chunks)


def read_chunks_sync(
    *,
    chroma_dir: str,
    source_file: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[StoredChunk, ...]:
    """同步离线入口；异步调用方应使用 ``await read_chunks()``。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            read_chunks(
                chroma_dir=chroma_dir,
                source_file=source_file,
                offset=offset,
                limit=limit,
            )
        )
    raise RuntimeError(
        "事件循环已运行，请使用 await knowledge.rag.offline.read_chunks(...)。"
    )


__all__ = ["read_chunks", "read_chunks_sync"]
