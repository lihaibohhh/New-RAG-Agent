"""Redis client resources with explicit lifecycle support."""
from __future__ import annotations

import logging
import threading

import redis as sync_redis
import redis.asyncio as aioredis

from react_agent.core.config import settings


logger = logging.getLogger(__name__)


class RedisClientManager:
    """Own one pair of lazy async/sync Redis connection pools."""

    def __init__(self, *, url: str, max_connections: int) -> None:
        self._url = url
        self._max_connections = max_connections
        self._async_pool: aioredis.ConnectionPool | None = None
        self._sync_pool: sync_redis.ConnectionPool | None = None
        self._lock = threading.Lock()

    def get_async(self) -> aioredis.Redis:
        if self._async_pool is None:
            with self._lock:
                if self._async_pool is None:
                    self._async_pool = aioredis.ConnectionPool.from_url(
                        self._url,
                        max_connections=self._max_connections,
                        decode_responses=False,
                    )
        return aioredis.Redis(connection_pool=self._async_pool)

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
        """Disconnect both pools and make the manager reusable."""
        with self._lock:
            async_pool = self._async_pool
            sync_pool = self._sync_pool
            self._async_pool = None
            self._sync_pool = None
        if async_pool is not None:
            await async_pool.disconnect(inuse_connections=True)
        if sync_pool is not None:
            sync_pool.disconnect(inuse_connections=True)


_default_manager = RedisClientManager(
    url=settings.redis.REDIS_URL,
    max_connections=settings.redis.REDIS_MAX_CONNECTIONS,
)


def get_async_redis() -> aioredis.Redis:
    """Compatibility provider for non-RAG process-level infrastructure."""
    return _default_manager.get_async()


def get_sync_redis() -> sync_redis.Redis:
    """Compatibility provider for non-RAG process-level infrastructure."""
    return _default_manager.get_sync()


async def ping_redis() -> bool:
    try:
        await get_async_redis().ping()
        return True
    except Exception as exc:
        logger.info("[Redis] 连接失败: %s，将降级为本地模式", exc)
        return False


async def close_default_redis() -> None:
    """Close the compatibility manager used outside explicit RAG scopes."""
    await _default_manager.close()


__all__ = [
    "RedisClientManager",
    "close_default_redis",
    "get_async_redis",
    "get_sync_redis",
    "ping_redis",
]
