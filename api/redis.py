"""FastAPI 限流与 Token 预算使用的 Redis 连接资源。"""

from __future__ import annotations

import threading

import redis.asyncio as async_redis

from api.settings import APISettings


class RedisClientManager:
    """惰性持有 API 进程使用的异步 Redis 连接池。"""

    def __init__(self, *, url: str, max_connections: int) -> None:
        self._url = url
        self._max_connections = max_connections
        self._pool: async_redis.ConnectionPool | None = None
        self._lock = threading.Lock()

    def get(self) -> async_redis.Redis:
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    self._pool = async_redis.ConnectionPool.from_url(
                        self._url,
                        max_connections=self._max_connections,
                        decode_responses=False,
                    )
        return async_redis.Redis(connection_pool=self._pool)

    async def close(self) -> None:
        """关闭连接池，并允许后续重新建立连接。"""
        with self._lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            await pool.disconnect(inuse_connections=True)


_default_manager: RedisClientManager | None = None
_default_manager_lock = threading.Lock()


def _get_default_manager() -> RedisClientManager:
    global _default_manager
    if _default_manager is None:
        with _default_manager_lock:
            if _default_manager is None:
                settings = APISettings()
                _default_manager = RedisClientManager(
                    url=settings.redis_url,
                    max_connections=settings.redis_max_connections,
                )
    return _default_manager


def get_async_redis() -> async_redis.Redis:
    """返回 API 进程共享的异步 Redis 客户端。"""
    return _get_default_manager().get()


async def close_default_redis() -> None:
    """关闭 API 进程共享的 Redis 连接池。"""
    global _default_manager
    with _default_manager_lock:
        manager = _default_manager
        _default_manager = None
    if manager is not None:
        await manager.close()


__all__ = ["RedisClientManager", "close_default_redis", "get_async_redis"]
