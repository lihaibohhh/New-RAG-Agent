"""将旧 Chroma 正文一次性迁移到中立 Chunk Store。"""
from __future__ import annotations

import asyncio
import logging
import threading

from knowledge.contracts import RagDocument, StoredChunk
from knowledge.infrastructure.retrieval.component_ports import (
    DocumentCorpusPort,
)
from knowledge.infrastructure.storage.chroma_knowledge_base import (
    validate_chroma_directory,
)
from knowledge.infrastructure.storage.sqlite_chunk_store import (
    SQLiteChunkStoreAdapter,
)


logger = logging.getLogger(__name__)


class ChromaCorpusReaderAdapter:
    """不加载 embedding，仅从旧 Chroma 集合导出逻辑正文。"""

    def __init__(
        self,
        *,
        chroma_dir: str,
        collection_name: str = "langchain",
    ) -> None:
        self._chroma_dir = chroma_dir
        self._collection_name = collection_name

    def list_documents(self) -> list[RagDocument]:
        import chromadb

        directory = validate_chroma_directory(self._chroma_dir)
        collection = chromadb.PersistentClient(path=str(directory)).get_collection(
            self._collection_name
        )
        result = collection.get(include=["documents", "metadatas"])
        ids = list(result.get("ids") or [])
        texts = list(result.get("documents") or [])
        metadatas = list(result.get("metadatas") or [])
        return [
            RagDocument(
                content=str(text or ""),
                metadata=(metadatas[index] if index < len(metadatas) else None),
                document_id=str(ids[index]) if index < len(ids) else None,
            )
            for index, text in enumerate(texts)
        ]


class MigratingChunkCorpusAdapter:
    """首次访问迁移旧库；完成标记与数据在同一 SQLite 事务提交。"""

    def __init__(
        self,
        *,
        store: SQLiteChunkStoreAdapter,
        legacy_source: DocumentCorpusPort,
        lock: threading.RLock | None = None,
    ) -> None:
        self._store = store
        self._legacy = legacy_source
        self._lock = lock or threading.RLock()

    def prepare(self) -> None:
        if self._store.snapshot_complete():
            return
        with self._lock:
            if self._store.snapshot_complete():
                return
            logger.info("[RAG] 开始从 Chroma 生成规范 Chunk Store 快照")
            documents = self._legacy.list_documents()
            self._store.replace_all(documents)
            logger.info("[RAG] Chunk Store 快照完成 documents=%s", len(documents))

    def list_documents(self) -> list[RagDocument]:
        self.prepare()
        return self._store.list_documents()

    async def list_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[StoredChunk]:
        await asyncio.to_thread(self.prepare)
        return await self._store.list_chunks(
            source_file=source_file,
            offset=offset,
            limit=limit,
        )


__all__ = ["ChromaCorpusReaderAdapter", "MigratingChunkCorpusAdapter"]
