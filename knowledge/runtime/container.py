"""Knowledge Server 的内部组合根。"""

from __future__ import annotations

from typing import Protocol

from knowledge.ingestion.runtime import LocalIngestionService
from knowledge.rag.runtime import LocalRagService
from knowledge.runtime.resources import SharedKnowledgeResources
from knowledge.services import IngestionService, RagService
from knowledge.settings import KnowledgeSettings


class _CacheInvalidationCoordinator(Protocol):
    """连接建库提交事件与 RAG 缓存失效，仅供组合根使用。"""

    async def invalidate(self) -> None: ...

    async def notify_index_changed(self) -> None: ...


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

    def _create_index_changed_handler(self) -> _CacheInvalidationCoordinator:
        return self._local_rag._create_cache_invalidator()

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


__all__ = ["KnowledgeServices", "create_knowledge_services"]
