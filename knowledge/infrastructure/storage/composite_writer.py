"""把同一批逻辑 Chunk 写入派生索引与规范语料层。"""
from __future__ import annotations

import threading

from knowledge.contracts import RagDocument
from knowledge.ports import VectorIndexWriterPort


class CompositeVectorIndexWriter:
    def __init__(
        self,
        *writers: VectorIndexWriterPort,
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
        documents: list[RagDocument],
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


__all__ = ["CompositeVectorIndexWriter"]
