"""与向量数据库无关的规范化 Chunk Store。"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path

from knowledge.contracts import ChunkMetadata, RagDocument, StoredChunk


class SQLiteChunkStoreAdapter:
    """保存可重建 BM25/向量索引的逻辑语料，不保存 embedding。"""

    def __init__(self, *, database_path: str) -> None:
        self._path = Path(database_path).expanduser().resolve()

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    document_id TEXT,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunk_store_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks(document_id);
                """
            )

    @staticmethod
    def _row(document: RagDocument, fallback_id: str) -> tuple[object, ...]:
        chunk_id = document.metadata.chunk_id or document.document_id or fallback_id
        metadata = document.metadata.to_dict()
        metadata.setdefault("chunk_id", chunk_id)
        return (
            chunk_id,
            document.content,
            json.dumps(
                metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ),
            document.document_id,
            time.time(),
        )

    def write(
        self,
        documents: list[RagDocument],
        *,
        batch_label: str,
        global_offset: int,
    ) -> None:
        del batch_label
        if not documents:
            return
        self.initialize()
        rows = [
            self._row(document, f"chunk_{global_offset + index}")
            for index, document in enumerate(documents)
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, content, metadata_json, document_id, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    content=excluded.content,
                    metadata_json=excluded.metadata_json,
                    document_id=excluded.document_id,
                    updated_at=excluded.updated_at
                """,
                rows,
            )

    def replace_all(self, documents: list[RagDocument]) -> None:
        """在单个 SQLite 事务中创建完整语料快照。"""
        self.initialize()
        rows = [
            self._row(document, f"chunk_{index}")
            for index, document in enumerate(documents)
        ]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM chunks")
            connection.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, content, metadata_json, document_id, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.execute(
                """
                INSERT INTO chunk_store_state(key, value)
                VALUES ('snapshot_complete', 'true')
                ON CONFLICT(key) DO UPDATE SET value='true'
                """
            )

    def snapshot_complete(self) -> bool:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM chunk_store_state WHERE key='snapshot_complete'"
            ).fetchone()
        return bool(row and row[0] == "true")

    def list_documents(self) -> list[RagDocument]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, content, metadata_json, document_id
                FROM chunks ORDER BY rowid
                """
            ).fetchall()
        return [
            RagDocument(
                content=str(content or ""),
                metadata=json.loads(metadata_json or "{}"),
                document_id=document_id or chunk_id,
            )
            for chunk_id, content, metadata_json, document_id in rows
        ]

    async def list_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[StoredChunk]:
        return await asyncio.to_thread(
            self._list_chunks_sync,
            source_file,
            offset,
            limit,
        )

    def _list_chunks_sync(
        self,
        source_file: str | None,
        offset: int,
        limit: int | None,
    ) -> list[StoredChunk]:
        self.initialize()
        query = "SELECT chunk_id, content, metadata_json FROM chunks"
        parameters: list[object] = []
        if source_file:
            query += (
                " WHERE json_extract(metadata_json, '$.source_file') = ?"
                " OR json_extract(metadata_json, '$.source') = ?"
            )
            parameters.extend((source_file, source_file))
        query += " ORDER BY rowid LIMIT ? OFFSET ?"
        parameters.extend((-1 if limit is None else max(1, limit), max(0, offset)))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            StoredChunk(
                chunk_id=str(chunk_id),
                content=str(content or ""),
                metadata=ChunkMetadata.from_mapping(json.loads(metadata_json or "{}")),
            )
            for chunk_id, content, metadata_json in rows
        ]


__all__ = ["SQLiteChunkStoreAdapter"]
