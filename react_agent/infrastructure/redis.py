"""具有显式生命周期的 Redis 客户端资源。"""
from __future__ import annotations

import logging
import threading

import redis as sync_redis
import redis.asyncio as async_redis

from react_agent.configuration.settings import settings


logger = logging.getLogger(__name__)


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
        """断开两个连接池，并允许实例后续重新建立连接。"""
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


def get_async_redis() -> async_redis.Redis:
    """为尚未迁入实例容器的调用方提供兼容客户端。"""
    return _default_manager.get_async()


def get_sync_redis() -> sync_redis.Redis:
    """为尚未迁入实例容器的调用方提供兼容客户端。"""
    return _default_manager.get_sync()


async def ping_redis() -> bool:
    try:
        await get_async_redis().ping()
        return True
    except Exception as exc:
        logger.info("[Redis] 连接失败: %s，将降级为本地模式", exc)
        return False


async def close_default_redis() -> None:
    """关闭进程级兼容连接池。"""
    await _default_manager.close()


__all__ = [
    "RedisClientManager",
    "close_default_redis",
    "get_async_redis",
    "get_sync_redis",
    "ping_redis",
]
