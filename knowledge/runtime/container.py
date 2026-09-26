"""Knowledge Server 的顶层组合根与跨模块能力视图。"""

from __future__ import annotations

from typing import cast

from knowledge.ingestion.runtime import LocalIngestionRuntime
from knowledge.rag.runtime import LocalRagRuntime
from knowledge.runtime.config import KnowledgeRuntimeConfig
from knowledge.runtime.internal_ports import CacheInvalidationCoordinatorPort
from knowledge.runtime.resources import SharedKnowledgeResources
from knowledge.runtime_ports import (
    IngestionRuntimePort,
    RagRuntimePort,
)


class KnowledgeRuntime:
    """连接自治模块，并持有二者共享资源的 Server 组合根。"""

    def __init__(self, config: KnowledgeRuntimeConfig) -> None:
        self._config = config
        self._resources = SharedKnowledgeResources(
            config.shared,
            chunk_store_path=config.storage.chunk_store_path,
        )
        self._rag_runtime = LocalRagRuntime(
            config.rag,
            chroma_dir=config.storage.chroma_db_path,
            device_provider=self._resources.get_device,
            embedding_provider=self._resources.get_embedding_model,
            chunk_store_provider=self._resources.get_chunk_store,
            knowledge_write_lock=self._resources.write_lock,
        )
        self._ingestion_runtime = LocalIngestionRuntime(
            config.ingestion,
            chroma_dir=config.storage.chroma_db_path,
            embedding_provider=self._resources.get_embedding_model,
            chunk_store_provider=self._resources.get_chunk_store,
            knowledge_write_lock=self._resources.write_lock,
            index_changed_provider=self._create_index_changed_handler,
        )

    @property
    def rag_runtime(self) -> RagRuntimePort:
        """返回不暴露建库能力的本地 RAG 视图。"""
        return self._rag_runtime

    @property
    def ingestion_runtime(self) -> IngestionRuntimePort:
        """返回不暴露查询能力的本地建库视图。"""
        return self._ingestion_runtime

    def _create_index_changed_handler(self) -> CacheInvalidationCoordinatorPort:
        return cast(
            CacheInvalidationCoordinatorPort,
            self._rag_runtime.create_cache_invalidator(),
        )

    async def close(self) -> None:
        """按模块后、共享资源后的顺序关闭完整对象图。"""
        try:
            await self._ingestion_runtime.close()
        finally:
            try:
                await self._rag_runtime.close()
            finally:
                await self._resources.close()


def create_knowledge_runtime(config: KnowledgeRuntimeConfig) -> KnowledgeRuntime:
    """创建由 Knowledge Server 独占的本地组合范围。"""
    return KnowledgeRuntime(config)


__all__ = ["KnowledgeRuntime", "create_knowledge_runtime"]
