"""Knowledge Server 内由 RAG 与建库共同使用的进程级资源。"""

from __future__ import annotations

import threading
from typing import Any, Literal

from knowledge.foundation.models import EmbeddingProviderAdapter
from knowledge.foundation.storage.sqlite_chunk_store import SQLiteChunkStoreAdapter
from knowledge.runtime.config import SharedResourceConfig
from knowledge.runtime.device import resolve_runtime_device


class SharedKnowledgeResources:
    """只管理跨模块共享资源，不组装任何 RAG 或建库用例。"""

    def __init__(
        self,
        config: SharedResourceConfig,
        *,
        chunk_store_path: str,
    ) -> None:
        self._config = config
        self._chunk_store_path = chunk_store_path
        self._device: Literal["cpu", "cuda"] | None = None
        self._embedding_provider: EmbeddingProviderAdapter | None = None
        self._chunk_store: SQLiteChunkStoreAdapter | None = None
        self._lock = threading.RLock()
        self.write_lock = threading.RLock()

    def get_embedding_model(self) -> Any:
        """惰性返回由两个模块共同复用的 Embedding 模型。"""
        return self._get_embedding_provider().get()

    def get_chunk_store(self) -> SQLiteChunkStoreAdapter:
        """惰性返回建库写入、RAG 读取共用的规范 Chunk Store。"""
        if self._chunk_store is not None:
            return self._chunk_store
        with self._lock:
            if self._chunk_store is None:
                self._chunk_store = SQLiteChunkStoreAdapter(
                    database_path=self._chunk_store_path,
                )
        return self._chunk_store

    def get_device(self) -> Literal["cpu", "cuda"]:
        """解析并缓存共享模型设备，供 Embedding 与 Reranker 一致使用。"""
        if self._device is not None:
            return self._device
        with self._lock:
            if self._device is None:
                self._device = resolve_runtime_device(self._config.requested_device)
        return self._device

    def _get_embedding_provider(self) -> EmbeddingProviderAdapter:
        if self._embedding_provider is not None:
            return self._embedding_provider
        with self._lock:
            if self._embedding_provider is None:
                self._embedding_provider = EmbeddingProviderAdapter(
                    model_name=self._config.embedding_model,
                    device=self.get_device(),
                )
        return self._embedding_provider

    async def close(self) -> None:
        provider = self._embedding_provider
        self._embedding_provider = None
        self._chunk_store = None
        if provider is not None:
            await provider.close()


__all__ = ["SharedKnowledgeResources"]
