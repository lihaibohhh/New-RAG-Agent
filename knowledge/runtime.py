"""Knowledge Server 的内部组合根。"""

from __future__ import annotations

import logging
import threading
from typing import Any, Literal

from knowledge.foundation.chunk_store import SQLiteChunkStoreAdapter
from knowledge.foundation.embedding import EmbeddingProviderAdapter
from knowledge.ingestion.runtime import LocalIngestionService
from knowledge.rag.runtime import LocalRagService
from knowledge.services import IngestionService, RagService
from knowledge.settings import KnowledgeSettings, SharedSettings


logger = logging.getLogger(__name__)
_VALID_DEVICES = {"auto", "cpu", "cuda"}


def resolve_runtime_device(requested: str = "auto") -> Literal["cpu", "cuda"]:
    """解析共享模型设备，并在 CUDA 不可用时回退到 CPU。"""
    configured = (requested or "auto").strip().lower()
    if configured not in _VALID_DEVICES:
        raise ValueError(
            "RAG_DEVICE 仅支持 auto、cpu 或 cuda，" f"当前值为 {configured!r}"
        )

    import torch

    cuda_available = torch.cuda.is_available()
    if configured == "cpu":
        return "cpu"
    if configured == "cuda" and not cuda_available:
        logger.warning("RAG_DEVICE=cuda，但当前运行环境无法使用 CUDA；已回退到 CPU")
        return "cpu"
    if configured == "cuda" or cuda_available:
        return "cuda"
    return "cpu"


class SharedKnowledgeResources:
    """管理 RAG 与建库共同复用的模型、Chunk Store 和写锁。"""

    def __init__(
        self,
        config: SharedSettings,
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
        return self._get_embedding_provider().get()

    def get_chunk_store(self) -> SQLiteChunkStoreAdapter:
        if self._chunk_store is not None:
            return self._chunk_store
        with self._lock:
            if self._chunk_store is None:
                self._chunk_store = SQLiteChunkStoreAdapter(
                    database_path=self._chunk_store_path,
                )
        return self._chunk_store

    def get_device(self) -> Literal["cpu", "cuda"]:
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


class KnowledgeServices:
    """连接建库与 RAG 服务，并统一管理共享资源。"""

    def __init__(self, config: KnowledgeSettings) -> None:
        self._config = config
        self._resources = SharedKnowledgeResources(
            config.shared,
            chunk_store_path=config.storage.chunk_store_path,
        )
        self._local_rag = LocalRagService(
            config.rag,
            chroma_dir=config.storage.chroma_db_path,
            device_provider=self._resources.get_device,
            embedding_provider=self._resources.get_embedding_model,
            chunk_store_provider=self._resources.get_chunk_store,
            knowledge_write_lock=self._resources.write_lock,
        )
        self.rag: RagService = self._local_rag
        self.ingestion: IngestionService = LocalIngestionService(
            config.ingestion,
            chroma_dir=config.storage.chroma_db_path,
            embedding_provider=self._resources.get_embedding_model,
            chunk_store_provider=self._resources.get_chunk_store,
            knowledge_write_lock=self._resources.write_lock,
            index_changed_provider=self._create_index_changed_handler,
        )

    def _create_index_changed_handler(self) -> Any:
        return self._local_rag.create_index_changed_handler()

    async def close(self) -> None:
        """按模块后、共享资源后的顺序关闭完整对象图。"""
        try:
            await self.ingestion.close()
        finally:
            try:
                await self.rag.close()
            finally:
                await self._resources.close()


def create_knowledge_services(config: KnowledgeSettings) -> KnowledgeServices:
    """创建由 Knowledge Server 独占的本地服务组合。"""
    return KnowledgeServices(config)


__all__ = [
    "KnowledgeServices",
    "SharedKnowledgeResources",
    "create_knowledge_services",
    "resolve_runtime_device",
]
