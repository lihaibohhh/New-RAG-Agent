"""知识库健康状态与缓存失效的管理服务。"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from knowledge.contracts import RagHealthStatus, StoredChunk
from knowledge.ports import KnowledgeBasePort, RagCacheInvalidatorPort


class RagAdminService:
    def __init__(
        self,
        *,
        cache_invalidator_provider: Callable[[], RagCacheInvalidatorPort],
        status_provider: Callable[[], dict[str, Any]],
        knowledge_base: KnowledgeBasePort | None = None,
        chunk_reader: KnowledgeBasePort | None = None,
    ) -> None:
        self._cache_invalidator_provider = cache_invalidator_provider
        self._status_provider = status_provider
        self._knowledge_base = knowledge_base
        self._chunk_reader = chunk_reader or knowledge_base

    async def health(self) -> RagHealthStatus:
        snapshot = self._status_provider()
        if self._knowledge_base is not None:
            knowledge_base = await self._knowledge_base.inspect()
            state = str(knowledge_base.get("status") or "unknown")
            return RagHealthStatus(
                ready=state == "knowledge_base_ready",
                state=state,
                details={
                    "knowledge_base": knowledge_base,
                    "runtime": snapshot,
                },
            )

        warmup = dict(snapshot.get("warmup_status") or {})
        state = str(warmup.get("state") or "not_started")
        return RagHealthStatus(
            ready=state == "done",
            state=state,
            details=snapshot,
        )

    async def read_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredChunk, ...]:
        """读取已入库原始片段，供评测和离线数据任务使用。"""
        if self._chunk_reader is None:
            raise RuntimeError("当前 RAG 管理服务未配置知识库读取端口")
        chunks = await self._chunk_reader.list_chunks(
            source_file=source_file,
            offset=offset,
            limit=limit,
        )
        return tuple(chunks)

    def read_chunks_sync(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredChunk, ...]:
        """同步脚本入口；异步调用方应使用 ``await read_chunks()``。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.read_chunks(
                    source_file=source_file,
                    offset=offset,
                    limit=limit,
                )
            )
        raise RuntimeError("事件循环已运行，请使用 await RagAdminService.read_chunks(...)。")

    async def invalidate(self, knowledge_base_id: str = "default") -> None:
        # 当前底层缓存仍是单知识库全局命名空间。显式拒绝伪多租户调用，
        # 避免调用方误以为只清除了某个知识库。
        if knowledge_base_id not in ("", "default"):
            raise ValueError("当前缓存实现仅支持 default 知识库全局失效")
        await self._cache_invalidator_provider().invalidate()
