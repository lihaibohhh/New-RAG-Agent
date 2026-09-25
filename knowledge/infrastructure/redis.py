"""Knowledge Runtime 专用、具有显式生命周期的 Redis 资源。"""
from __future__ import annotations

import threading

import redis as sync_redis
import redis.asyncio as async_redis


class RedisClientManager:
    """持有一组惰性同步与异步 Redis 连接池。"""

    def __init__(self, *, url: str, max_connections: int) -> None:
        self._url = url
        self._max_connections = max_connections
        self._async_pool: async_redis.ConnectionPool | None = None
        self._sync_pool: sync_redis.ConnectionPool | None = None
        self._lock = threading.Lock()

    def get_async(self) -> async_redis.Redis:
        if self._async_pool is None:
            with self._lock:
                if self._async_pool is None:
                    self._async_pool = async_redis.ConnectionPool.from_url(
                        self._url,
                        max_connections=self._max_connections,
                        decode_responses=False,
                    )
        return async_redis.Redis(connection_pool=self._async_pool)

    def get_sync(self) -> sync_redis.Redis:
        if self._sync_pool is None:
            with self._lock:
                if self._sync_pool is None:
                    self._sync_pool = sync_redis.ConnectionPool.from_url(
                        self._url,
                        max_connections=self._max_connections,
                        decode_responses=False,
                    )
        return sync_redis.Redis(connection_pool=self._sync_pool)

    async def close(self) -> None:
        with self._lock:
            async_pool = self._async_pool
            sync_pool = self._sync_pool
            self._async_pool = None
            self._sync_pool = None
        if async_pool is not None:
            await async_pool.disconnect(inuse_connections=True)
        if sync_pool is not None:
            sync_pool.disconnect(inuse_connections=True)


__all__ = ["RedisClientManager"]
