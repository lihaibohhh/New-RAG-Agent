"""BM25 索引的 Redis 持久化适配器。"""
from __future__ import annotations

import hashlib
import hmac
import logging
import pickle
from collections.abc import Callable
from typing import Any

from react_agent.rag.infrastructure.retrieval.bm25_tokenizer import (
    bm25_tokenizer_fingerprint,
)
from react_agent.infrastructure.redis import get_sync_redis


logger = logging.getLogger(__name__)
_INDEX_KEY = "rag:bm25_index"
_COUNT_KEY = "rag:bm25_doc_count"
_HMAC_KEY = "rag:bm25_hmac"
_VERSION_KEY = "rag:bm25_index_version"


class RedisBm25Repository:
    """使用 HMAC 校验后持久化第三方 BM25 后端对象。"""

    def __init__(
        self,
        *,
        hmac_secret: str,
        ttl: int,
        redis_provider: Callable[[], Any] = get_sync_redis,
    ) -> None:
        self._secret = hmac_secret.encode("utf-8") if hmac_secret else b""
        self._ttl = ttl
        self._redis_provider = redis_provider

    def _sign(self, data: bytes) -> str:
        return hmac.new(self._secret, data, hashlib.sha256).hexdigest()

    def load(self) -> Any | None:
        if not self._secret:
            logger.info("[RAG] BM25_HMAC_SECRET 未配置，跳过 Redis 索引缓存")
            return None
        try:
            redis = self._redis_provider()
            data = redis.get(_INDEX_KEY)
            if not data:
                return None
            raw_version = redis.get(_VERSION_KEY)
            version = (
                raw_version.decode("utf-8")
                if isinstance(raw_version, bytes)
                else str(raw_version or "")
            )
            expected_version = bm25_tokenizer_fingerprint()
            if version != expected_version:
                logger.info(
                    "[RAG] BM25 Redis 缓存分词版本不兼容，将重建 "
                    "cached=%s expected=%s",
                    version or "legacy",
                    expected_version,
                )
                return None
            raw_signature = redis.get(_HMAC_KEY)
            if not raw_signature:
                logger.warning("[RAG] BM25 Redis 缓存缺少签名，拒绝加载")
                return None
            signature = (
                raw_signature.decode("utf-8")
                if isinstance(raw_signature, bytes)
                else str(raw_signature)
            )
            if not hmac.compare_digest(self._sign(data), signature):
                logger.warning("[RAG] BM25 Redis 缓存签名验证失败，拒绝加载")
                return None
            backend = pickle.loads(data)
            raw_count = redis.get(_COUNT_KEY)
            count = int(raw_count.decode("utf-8")) if raw_count else 0
            logger.info("[RAG] BM25 索引从 Redis 加载 documents=%s", count)
            return backend
        except Exception as exc:
            logger.warning("[RAG] BM25 Redis 读取失败，将重建索引: %s", exc)
            return None

    def save(self, backend: Any, document_count: int) -> None:
        if not self._secret:
            return
        try:
            data = pickle.dumps(backend)
            pipeline = self._redis_provider().pipeline()
            pipeline.set(_INDEX_KEY, data, ex=self._ttl)
            pipeline.set(_COUNT_KEY, str(document_count), ex=self._ttl)
            pipeline.set(_HMAC_KEY, self._sign(data), ex=self._ttl)
            pipeline.set(
                _VERSION_KEY,
                bm25_tokenizer_fingerprint(),
                ex=self._ttl,
            )
            pipeline.execute()
            logger.info(
                "[RAG] BM25 索引写入 Redis documents=%s ttl=%s",
                document_count,
                self._ttl,
            )
        except Exception as exc:
            logger.warning("[RAG] BM25 Redis 写入失败，仅保留内存索引: %s", exc)

    def clear(self) -> None:
        try:
            self._redis_provider().delete(
                _INDEX_KEY,
                _COUNT_KEY,
                _HMAC_KEY,
                _VERSION_KEY,
            )
        except Exception as exc:
            raise RuntimeError("BM25 Redis 缓存清理失败") from exc


__all__ = ["RedisBm25Repository"]
