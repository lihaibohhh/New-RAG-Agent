"""RAG 缓存与候选索引的统一失效适配器。"""
from __future__ import annotations

import asyncio
from collections.abc import Callable

from knowledge.rag.ports import (
    Bm25CachePort,
    HybridRetrieverPort,
    SemanticCachePort,
)


class RagCacheInvalidator:
    """分别清理 BM25 索引缓存和当前版本的查询结果缓存。"""

    def __init__(
        self,
        *,
        bm25_cache: Bm25CachePort,
        query_cache: SemanticCachePort,
        retriever_provider: Callable[[], HybridRetrieverPort | None],
    ) -> None:
        self._bm25_cache = bm25_cache
        self._query_cache = query_cache
        self._retriever_provider = retriever_provider

    async def invalidate(self) -> None:
        retriever = self._retriever_provider()
        if retriever is None:
            await asyncio.to_thread(self._bm25_cache.clear)
        else:
            await retriever.invalidate()
        await self._query_cache.clear()

    async def notify_index_changed(self) -> None:
        """响应建库提交事件；保留 invalidate() 供管理 API 显式调用。"""
        await self.invalidate()


__all__ = ["RagCacheInvalidator"]
