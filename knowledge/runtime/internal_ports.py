"""仅供 Knowledge Runtime 组合根使用的跨模块端口。"""

from __future__ import annotations

from typing import Protocol


class CacheInvalidationCoordinatorPort(Protocol):
    """连接建库提交事件与 RAG 缓存失效，不对外公开。"""

    async def invalidate(self) -> None: ...

    async def notify_index_changed(self) -> None: ...


__all__ = ["CacheInvalidationCoordinatorPort"]
