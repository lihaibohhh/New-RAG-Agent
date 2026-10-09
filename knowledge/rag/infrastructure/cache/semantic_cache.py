"""RAG 私有的 Redis 语义缓存适配器。"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Optional

import numpy as np

from knowledge.contracts import KnowledgeDocument
from knowledge.rag.infrastructure.cache.codec import (
    deserialize_documents,
    deserialize_vector,
    serialize_documents,
    serialize_vector,
)


logger = logging.getLogger(__name__)

_CACHE_VERSION = "v2"
_EXACT_PREFIX = f"rag:{_CACHE_VERSION}:exact:"          # Tier-1 精确匹配结果
_SEM_VEC_PREFIX = f"rag:{_CACHE_VERSION}:semantic:vec:" # Tier-2 查询向量
_SEM_RES_PREFIX = f"rag:{_CACHE_VERSION}:semantic:res:" # Tier-2 检索结果
_SEM_INDEX_KEY = f"rag:{_CACHE_VERSION}:semantic:index" # Tier-2 索引（HASH）

_NO_RESULT_TTL = 300  # 无结果短缓存 5 分钟
_MAX_INDEX_SIZE = 2000
_REDIS_PROVIDER: ContextVar[Callable[[], Any] | None] = ContextVar(
    "rag_semantic_cache_redis_provider",
    default=None,
)
_EMBEDDING_PROVIDER: ContextVar[Callable[[], Any] | None] = ContextVar(
    "rag_semantic_cache_embedding_provider",
    default=None,
)


def _get_redis():
    provider = _REDIS_PROVIDER.get()
    if provider is None:
        raise RuntimeError("Semantic Cache 必须由 Runtime 注入 redis_provider")
    return provider()


class RedisSemanticCacheAdapter:
    """Redis 两级语义缓存的端口实现。"""

    def __init__(
        self,
        *,
        redis_provider: Callable[[], Any],
        embedding_provider: Callable[[], Any],
        ttl: int = 3_600,
        threshold: float = 0.95,
        min_score: float = 0.5,
    ) -> None:
        self._redis_provider = redis_provider
        self._embedding_provider = embedding_provider
        self._ttl = max(1, int(ttl))
        self._threshold = float(threshold)
        self._min_score = float(min_score)

    async def get(self, query: str) -> list[KnowledgeDocument] | None:
        redis_token = _REDIS_PROVIDER.set(self._redis_provider)
        embedding_token = _EMBEDDING_PROVIDER.set(self._embedding_provider)
        try:
            return await get_cached_result(query, threshold=self._threshold)
        finally:
            _EMBEDDING_PROVIDER.reset(embedding_token)
            _REDIS_PROVIDER.reset(redis_token)

    async def set(
        self,
        query: str,
        documents: list[KnowledgeDocument],
        *,
        top_score: float,
    ) -> None:
        redis_token = _REDIS_PROVIDER.set(self._redis_provider)
        embedding_token = _EMBEDDING_PROVIDER.set(self._embedding_provider)
        try:
            await set_cached_result(
                query,
                documents,
                top_score=top_score,
                ttl=self._ttl,
                min_score=self._min_score,
            )
        finally:
            _EMBEDDING_PROVIDER.reset(embedding_token)
            _REDIS_PROVIDER.reset(redis_token)

    async def clear(self) -> None:
        token = _REDIS_PROVIDER.set(self._redis_provider)
        try:
            await clear_query_cache()
        finally:
            _REDIS_PROVIDER.reset(token)


async def _embed(query: str) -> np.ndarray:
    """异步嵌入查询字符串（在线程池中运行，不阻塞事件循环）"""
    provider = _EMBEDDING_PROVIDER.get()
    if provider is None:
        raise RuntimeError("Semantic Cache 必须由 Runtime 注入 embedding_provider")
    embedder = await asyncio.to_thread(provider)
    vec = await asyncio.to_thread(embedder.embed_query, query)
    return np.array(vec, dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


def _query_hash(query: str) -> str:
    """生成不可逆查询摘要，避免在 Redis key 中暴露查询原文。"""
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()


def _decode(v) -> str:
    """Redis bytes key 解码（decode_responses=False 模式下需要手动 decode）"""
    return v.decode("utf-8") if isinstance(v, bytes) else v


# ── Tier-1: 精确匹配 ──────────────────────────────────────────────────────────
async def _exact_get(query: str) -> Optional[list[KnowledgeDocument]]:
    try:
        r = _get_redis()
        data = await r.get(_EXACT_PREFIX + _query_hash(query))
        if data:
            logger.info("[SemanticCache] ✅ Tier-1 精确命中: %s...", query[:40])
            return deserialize_documents(data)
    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-1 读取失败: {e}")
    return None


async def _exact_set(query: str, docs: list[KnowledgeDocument], ttl: int) -> None:
    try:
        r = _get_redis()
        await r.set(
            _EXACT_PREFIX + _query_hash(query),
            serialize_documents(docs),
            ex=ttl,
        )
    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-1 写入失败: {e}")


# ── Tier-2: 语义相似匹配 ──────────────────────────────────────────────────────
async def _semantic_get(
    query_vec: np.ndarray,
    *,
    threshold: float,
) -> Optional[list[KnowledgeDocument]]:
    """
    批量拉取所有已缓存向量 → 线性余弦相似度扫描 → 返回最优命中结果。

    规模估算（_MAX_INDEX_SIZE = 2000）：
      bge-small-zh-v1.5 向量维度 512，float32 = 2 KB/条
      2000 条 × 2 KB = ~4 MB，单次 MGET 完全可接受。
      向量已过期（mget 返回 None）的条目会被静默跳过，不影响正确性。
    """
    try:
        r = _get_redis()

        raw_ids = await r.hkeys(_SEM_INDEX_KEY)
        if not raw_ids:
            return None

        entry_ids = [_decode(eid) for eid in raw_ids]
        vec_keys  = [_SEM_VEC_PREFIX + eid for eid in entry_ids]
        raw_vecs  = await r.mget(*vec_keys)

        best_score, best_id = -1.0, None
        for eid, raw_vec in zip(entry_ids, raw_vecs):
            if raw_vec is None:
                # 向量 key 已 TTL 过期，对应 index 条目变为孤儿，跳过即可
                continue
            try:
                cached_vec = deserialize_vector(
                    raw_vec,
                    expected_size=query_vec.size,
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "[SemanticCache] ⚠️ 跳过损坏向量 entry_id=%s: %s",
                    eid,
                    exc,
                )
                continue
            score = _cosine_sim(query_vec, cached_vec)
            if score > best_score:
                best_score, best_id = score, eid

        if best_score >= threshold and best_id is not None:
            raw_res = await r.get(_SEM_RES_PREFIX + best_id)
            if raw_res:
                logger.info(
                    f"[SemanticCache] ✅ Tier-2 语义命中 "
                    f"(sim={best_score:.4f} ≥ {threshold})"
                )
                return deserialize_documents(raw_res)

    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-2 读取失败: {e}")
    return None


async def _semantic_set(
    entry_id: str,
    query_vec: np.ndarray,
    docs: list[KnowledgeDocument],
    ttl: int,
    query: str,
) -> None:
    """
    写入语义索引，含两阶段容量管理：
      1. 清理孤儿条目（向量 key 已 TTL 过期，但 index HASH 仍有记录）
      2. 若清完后仍超限，移除最早的 10% 条目
    """
    try:
        r = _get_redis()
        index_size = await r.hlen(_SEM_INDEX_KEY)

        if index_size >= _MAX_INDEX_SIZE:
            all_ids = [_decode(eid) for eid in await r.hkeys(_SEM_INDEX_KEY)]
            exists = await r.mget(*[_SEM_VEC_PREFIX + eid for eid in all_ids])
            orphan_ids = [eid for eid, v in zip(all_ids, exists) if v is None]

            if orphan_ids:
                pipe = r.pipeline()
                for oid in orphan_ids:
                    pipe.hdel(_SEM_INDEX_KEY, oid)
                    pipe.delete(_SEM_RES_PREFIX + oid)
                await pipe.execute()
                logger.info(f"[SemanticCache] 🗑️ 清理孤儿条目 {len(orphan_ids)} 个")
                index_size -= len(orphan_ids)

            # 清完孤儿仍然超限 → 移除最早的 10%
            if index_size >= _MAX_INDEX_SIZE:
                n_remove  = max(1, _MAX_INDEX_SIZE // 10)
                evict_ids = all_ids[:n_remove]
                pipe = r.pipeline()
                for oid in evict_ids:
                    pipe.hdel(_SEM_INDEX_KEY, oid)
                    pipe.delete(_SEM_VEC_PREFIX + oid)
                    pipe.delete(_SEM_RES_PREFIX + oid)
                await pipe.execute()
                logger.info(f"[SemanticCache] 🗑️ 容量超限，淘汰最旧 {len(evict_ids)} 条")

        # 原子写入向量、结果、索引记录（三条 Pipeline，TTL 各自独立）
        pipe = r.pipeline()
        pipe.set(_SEM_VEC_PREFIX + entry_id, serialize_vector(query_vec), ex=ttl)
        pipe.set(_SEM_RES_PREFIX + entry_id, serialize_documents(docs), ex=ttl)
        pipe.hset(_SEM_INDEX_KEY, entry_id, query[:60])  # 预览文本仅供调试
        await pipe.execute()

    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-2 写入失败: {e}")


async def get_cached_result(
    query: str,
    *,
    threshold: float,
) -> Optional[list[KnowledgeDocument]]:
    """
    两级查找（接口签名与原版完全兼容）：

      Tier-1  精确 MD5 匹配 —— 同一 query 字符串，O(1)，无 Embedding 开销
      Tier-2  语义向量匹配 —— 不同措辞但语义相近的 query，使用 Runtime 注入阈值
    """
    # Tier-1: 零 Embedding 开销的快速路径
    result = await _exact_get(query)
    if result is not None:
        return result

    # Tier-2: 语义匹配
    try:
        query_vec = await _embed(query)
        result = await _semantic_get(query_vec, threshold=threshold)
        if result is not None:
            logger.info("Redis 语义缓存生效 ✅")
            return result
    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-2 查询异常，降级跳过: {e}")

    return None


async def set_cached_result(
    query: str,
    docs: list[KnowledgeDocument],
    top_score: float = 0.0,
    *,
    ttl: int,
    min_score: float,
) -> None:
    """
    写入双级缓存（接口签名与原版完全兼容）：

      空结果      → 仅 Tier-1 短缓存（_NO_RESULT_TTL），不浪费向量索引槽位
      低置信度    → 两级均跳过（top_score < CACHE_MIN_SCORE）
      正常结果    → Tier-1 精确缓存 + Tier-2 语义索引同步写入
      Tier-2 失败 → 降级保留 Tier-1，整体不抛出异常
    """
    if not docs:
        await _exact_set(query, docs, _NO_RESULT_TTL)
        logger.info(f"[SemanticCache] 💾 空结果短缓存 (TTL={_NO_RESULT_TTL}s): {query[:40]}...")
        return

    if top_score < min_score:
        logger.info(
            f"[SemanticCache] ⏭️ 跳过缓存 "
            f"(score={top_score:.3f} < {min_score}): {query[:40]}..."
        )
        return

    # Tier-1 精确写入
    await _exact_set(query, docs, ttl)

    # Tier-2 语义写入（entry_id 与 Tier-1 使用同一 md5，相同 query 自然去重）
    try:
        query_vec = await _embed(query)
        entry_id = _query_hash(query)
        await _semantic_set(entry_id, query_vec, docs, ttl, query)
        logger.info(
            f"[SemanticCache] 💾 Tier-1+2 写入完成 "
            f"(score={top_score:.3f}, TTL={ttl}s): {query[:40]}..."
        )
    except Exception as e:
        logger.warning(f"[SemanticCache] ⚠️ Tier-2 写入失败，Tier-1 已写入: {e}")


async def clear_query_cache() -> None:
    """清除当前版本的精确与语义查询缓存，不触碰 BM25 索引。"""
    r = _get_redis()
    keys = [k async for k in r.scan_iter(f"rag:{_CACHE_VERSION}:*")]
    if keys:
        await r.delete(*keys)
        logger.info(
            "[SemanticCache] 🗑️ 已清除 %s 个查询缓存键（精确 + 语义）",
            len(keys),
        )


__all__ = ["RedisSemanticCacheAdapter", "clear_query_cache"]
