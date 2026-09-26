"""RAG 管理用例使用的 Chroma 诊断与只读访问适配器。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from knowledge.contracts import ChunkMetadata, StoredChunk
from knowledge.rag.infrastructure.storage.chroma import validate_chroma_directory


class ChromaKnowledgeBaseAdapter:
    """将 Chroma 原生集合结构隔离在 RAG 管理适配器内。"""

    def __init__(
        self,
        *,
        chroma_dir: str,
        collection_name: str = "langchain",
    ) -> None:
        if not chroma_dir:
            raise ValueError("chroma_dir 不能为空")
        self._chroma_dir = chroma_dir
        self._collection_name = collection_name

    def _resolved_dir(self) -> str:
        return str(Path(self._chroma_dir))

    def _collection(self) -> Any:
        import chromadb

        chroma_dir = validate_chroma_directory(self._resolved_dir())
        client = chromadb.PersistentClient(path=str(chroma_dir))
        return client.get_collection(self._collection_name)

    async def inspect(self) -> dict[str, Any]:
        collection = await asyncio.to_thread(self._collection)
        chunk_count = await asyncio.to_thread(collection.count)
        return {
            "status": "knowledge_base_ready",
            "chunk_count": int(chunk_count),
            "chroma_dir": self._resolved_dir(),
            "collection": self._collection_name,
            "embedding_loaded": False,
            "note": "轻量健康检查；未加载嵌入模型或精排器",
        }

    async def list_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[StoredChunk]:
        collection = await asyncio.to_thread(self._collection)
        kwargs: dict[str, Any] = {
            "include": ["documents", "metadatas"],
        }
        if source_file:
            kwargs["where"] = {"source": source_file}
        if offset:
            kwargs["offset"] = max(0, int(offset))
        if limit is not None:
            kwargs["limit"] = max(1, int(limit))
        result = await asyncio.to_thread(collection.get, **kwargs)

        ids = list(result.get("ids") or [])
        documents = list(result.get("documents") or [])
        metadatas = list(result.get("metadatas") or [])
        chunks: list[StoredChunk] = []
        for index, content in enumerate(documents):
            metadata = ChunkMetadata.from_mapping(
                metadatas[index] if index < len(metadatas) else None
            )
            chunk_id = str(
                ids[index] if index < len(ids) else metadata.chunk_id or index
            )
            chunks.append(
                StoredChunk(
                    chunk_id=chunk_id,
                    content=str(content or ""),
                    metadata=metadata,
                )
            )
        return chunks


__all__ = ["ChromaKnowledgeBaseAdapter"]
