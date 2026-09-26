"""RAG 查询缓存适配器。"""

from knowledge.rag.infrastructure.cache.semantic_cache import (
    RedisSemanticCacheAdapter,
)

__all__ = ["RedisSemanticCacheAdapter"]
