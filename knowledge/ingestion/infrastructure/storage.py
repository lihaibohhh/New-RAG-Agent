"""建库使用的增量清单、Chroma 写入和多存储协调。"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from knowledge.contracts import KnowledgeDocument
from knowledge.ingestion.observability import timer


logger = logging.getLogger(__name__)


class JsonIngestionManifestAdapter:
    """持久化文件哈希，并通过同目录原子替换避免半写文件。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"建库指纹清单必须是 JSON 对象：{self.path}")
        return {str(key): str(value) for key, value in payload.items()}

    def save(self, record: dict[str, str]) -> None:
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(record, stream, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
                temp_path = Path(stream.name)
            os.replace(temp_path, target)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()


class ChromaVectorWriterAdapter:
    """初始化 Chroma，并批量写入解析后的文档。"""

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

        logger.info("[INGESTION] 加载 Runtime 注入的 embedding 模型")
        with timer("T3_model_load"):
            self._embeddings = self._embedding_provider()
            self._vectorstore = Chroma(
                persist_directory=self._chroma_dir,
                embedding_function=self._embeddings,
            )

    def write(
        self,
        documents: list[KnowledgeDocument],
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


class CompositeVectorIndexWriter:
    """在同一把写锁下，将批次写入全部知识索引。"""

    def __init__(
        self,
        *writers: Any,
        lock: threading.RLock | None = None,
    ) -> None:
        self._writers = writers
        self._lock = lock or threading.RLock()

    def initialize(self) -> None:
        with self._lock:
            for writer in self._writers:
                writer.initialize()

    def write(
        self,
        documents: list[KnowledgeDocument],
        *,
        batch_label: str,
        global_offset: int,
    ) -> None:
        with self._lock:
            for writer in self._writers:
                writer.write(
                    documents,
                    batch_label=batch_label,
                    global_offset=global_offset,
                )


__all__ = [
    "ChromaVectorWriterAdapter",
    "CompositeVectorIndexWriter",
    "JsonIngestionManifestAdapter",
]
