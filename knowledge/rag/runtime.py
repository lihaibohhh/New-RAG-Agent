"""RAG 模块拥有的本地对象图与生命周期。"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from knowledge.contracts import (
    RagHealthStatus,
    StoredChunk,
)
from knowledge.rag.application import RetrievalApplicationService
from knowledge.rag.contracts import (
    ChunkStorePort,
    WarmupServicePort,
)
from knowledge.rag.infrastructure.bm25 import (
    LazyBm25CandidateRetrieverAdapter,
    RedisBm25Repository,
)
from knowledge.rag.infrastructure.chroma import ChromaKnowledgeBaseAdapter
from knowledge.rag.infrastructure.corpus import (
    ChromaCorpusReaderAdapter,
    MigratingChunkCorpusAdapter,
)
from knowledge.rag.infrastructure.redis import RedisClientManager
from knowledge.rag.infrastructure.reranker import RerankerProviderAdapter
from knowledge.rag.infrastructure.semantic_cache import RedisSemanticCacheAdapter
from knowledge.rag.infrastructure.vector import ChromaCandidateRetriever
from knowledge.settings import RagSettings

logger = logging.getLogger(__name__)
WarmupState = Literal["not_started", "running", "done", "error", "cancelled"]


class WarmupStatusSnapshot(TypedDict):
    state: WarmupState
    started_at: float | None
    finished_at: float | None
    timings: dict[str, float]
    error: str | None


def _initial_warmup_status() -> WarmupStatusSnapshot:
    return {
        "state": "not_started",
        "started_at": None,
        "finished_at": None,
        "timings": {},
        "error": None,
    }


class RagWarmupManager:
    """管理 RAG 预热顺序、幂等、状态和后台任务生命周期。"""

    def __init__(
        self,
        retrieval_service_provider: Callable[[], WarmupServicePort],
    ) -> None:
        self._retrieval_service_provider = retrieval_service_provider
        self._task: asyncio.Task | None = None
        self._status = _initial_warmup_status()

    @staticmethod
    def _elapsed(started: float) -> float:
        return round(time.perf_counter() - started, 4)

    @staticmethod
    def _safe_wait_seconds(value: Any, default: int = 20) -> float:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            seconds = float(default)
        return max(0.0, min(seconds, 120.0))

    def task_state(self) -> str:
        if self._task is None:
            return "none"
        return "done" if self._task.done() else "running"

    def _snapshot(self) -> WarmupStatusSnapshot:
        return {
            "state": self._status["state"],
            "started_at": self._status["started_at"],
            "finished_at": self._status["finished_at"],
            "timings": dict(self._status["timings"]),
            "error": self._status["error"],
        }

    def get_status(self) -> dict[str, Any]:
        return {
            "task_state": self.task_state(),
            "warmup_status": self._snapshot(),
        }

    async def _run(self) -> None:
        current_task = asyncio.current_task()
        total_started = time.perf_counter()
        self._status.update(
            state="running",
            started_at=time.time(),
            finished_at=None,
            timings={},
            error=None,
        )
        logger.info("[RAG] 查询服务预热开始 pid=%s", os.getpid())
        try:
            result = await self._retrieval_service_provider().warmup()
            timings = dict(result.timings)
            timings["total"] = self._elapsed(total_started)
            if self._task is current_task:
                self._status.update(
                    state="done",
                    finished_at=time.time(),
                    timings=timings,
                    error=None,
                )
                logger.info(
                    "[RAG] 查询服务预热完成 pid=%s timings=%s",
                    os.getpid(),
                    timings,
                )
        except asyncio.CancelledError:
            if self._task is current_task:
                self._status.update(
                    state="cancelled",
                    finished_at=time.time(),
                    timings={"total": self._elapsed(total_started)},
                    error="CancelledError",
                )
            raise
        except Exception as exc:
            if self._task is current_task:
                self._status.update(
                    state="error",
                    finished_at=time.time(),
                    timings={"total": self._elapsed(total_started)},
                    error=f"{type(exc).__name__}: {exc}",
                )
                logger.exception("[RAG] 查询服务预热失败")

    def start(self, force: bool = False) -> dict[str, Any]:
        current_task_state = self.task_state()
        status = self._snapshot()
        if current_task_state == "running":
            return {
                "stage": "already_running",
                "task_state": current_task_state,
                "warmup_status": status,
            }
        if status["state"] == "done" and not force:
            return {
                "stage": "already_done",
                "task_state": current_task_state,
                "warmup_status": status,
            }

        self._task = asyncio.create_task(
            self._run(),
            name="rag_service_warmup",
        )
        return {"stage": "started", "task_state": self.task_state()}

    @staticmethod
    def _ready_result(
        status: WarmupStatusSnapshot,
        *,
        stage: str,
        waited_seconds: float,
    ) -> dict[str, Any] | None:
        state = status["state"]
        if state == "done":
            return {
                "ready": True,
                "stage": stage,
                "waited_seconds": waited_seconds,
                "warmup_status": status,
            }
        if state in {"error", "cancelled"}:
            return {
                "ready": False,
                "stage": f"warmup_{state}",
                "retry_after_seconds": None,
                "waited_seconds": waited_seconds,
                "warmup_status": status,
            }
        return None

    async def ensure_ready(
        self,
        wait_seconds: int | float = 20,
    ) -> dict[str, Any]:
        wait_budget = self._safe_wait_seconds(wait_seconds)
        ready = self._ready_result(
            self._snapshot(),
            stage="ready",
            waited_seconds=0.0,
        )
        if ready is not None:
            return ready

        if self.task_state() != "running":
            self.start(force=False)

        started = time.perf_counter()
        while True:
            status = self._snapshot()
            ready = self._ready_result(
                status,
                stage="ready_after_wait",
                waited_seconds=self._elapsed(started),
            )
            if ready is not None:
                return ready

            elapsed = time.perf_counter() - started
            if elapsed >= wait_budget:
                return {
                    "ready": False,
                    "stage": "warming_up",
                    "retry_after_seconds": 5,
                    "waited_seconds": round(elapsed, 4),
                    "warmup_status": status,
                }
            await asyncio.sleep(min(0.5, wait_budget - elapsed))

    async def close(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._task = None

    def reset_for_testing(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
        self._task = None
        self._status = _initial_warmup_status()


async def read_chunks(
    *,
    chroma_dir: str,
    source_file: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[StoredChunk, ...]:
    """离线读取指定 Chroma，不创建完整 Knowledge Runtime。"""
    reader = ChromaKnowledgeBaseAdapter(chroma_dir=chroma_dir)
    chunks = await reader.list_chunks(
        source_file=source_file,
        offset=offset,
        limit=limit,
    )
    return tuple(chunks)


def read_chunks_sync(
    *,
    chroma_dir: str,
    source_file: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[StoredChunk, ...]:
    """同步离线入口；异步调用方应使用 ``await read_chunks()``。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            read_chunks(
                chroma_dir=chroma_dir,
                source_file=source_file,
                offset=offset,
                limit=limit,
            )
        )
    raise RuntimeError(
        "事件循环已运行，请使用 await knowledge.rag.runtime.read_chunks(...)。"
    )


class LocalRagService(RetrievalApplicationService):
    """可直接检索的应用服务，同时管理评测、运维和资源生命周期。"""

    def __init__(
        self,
        config: RagSettings,
        *,
        chroma_dir: str,
        device_provider: Callable[[], Literal["cpu", "cuda"]],
        embedding_provider: Callable[[], Any],
        chunk_store_provider: Callable[[], ChunkStorePort],
        knowledge_write_lock: threading.RLock,
    ) -> None:
        profile = config.tuning.profile(device_provider())
        logger.info(
            "[RAG] runtime profile device=%s rerank_candidates=%s "
            "rerank_top_n=%s reranker_concurrency=%s reranker_batch_size=%s",
            profile.device,
            profile.rerank_candidates,
            profile.rerank_top_n,
            profile.reranker_concurrency,
            profile.reranker_batch_size,
        )
        self._redis_resources = RedisClientManager(
            url=config.redis.url,
            max_connections=config.redis.max_connections,
        )
        self._query_cache = RedisSemanticCacheAdapter(
            redis_provider=self._redis_resources.get_async,
            embedding_provider=embedding_provider,
            ttl=config.redis.semantic_cache_ttl,
            threshold=config.redis.semantic_cache_threshold,
            min_score=config.redis.semantic_cache_min_score,
        )
        bm25_repository = RedisBm25Repository(
            hmac_secret=config.redis.bm25_hmac_secret,
            ttl=config.redis.bm25_index_ttl,
            redis_provider=self._redis_resources.get_sync,
        )
        self._chunk_corpus = MigratingChunkCorpusAdapter(
            store=chunk_store_provider(),
            legacy_source=ChromaCorpusReaderAdapter(chroma_dir=chroma_dir),
            lock=knowledge_write_lock,
        )
        self._vector_retriever = ChromaCandidateRetriever(
            chroma_dir=chroma_dir,
            top_k=10,
            embedding_provider=embedding_provider,
        )
        self._bm25_retriever = LazyBm25CandidateRetrieverAdapter(
            repository=bm25_repository,
            corpus_reader=self._chunk_corpus,
            top_k=10,
        )
        self._reranker = RerankerProviderAdapter(
            device=profile.device,
            inference_concurrency=profile.reranker_concurrency,
            batch_size=profile.reranker_batch_size,
            model_name=config.reranker.model,
            threshold=config.reranker.threshold,
            debug=config.reranker.debug,
        )
        super().__init__(
            cache=self._query_cache,
            vector_retriever=self._vector_retriever,
            bm25_retriever=self._bm25_retriever,
            reranker=self._reranker,
            max_content_chars=config.tuning.max_content_chars,
            source_top_k=10,
            rrf_rank_constant=60,
            rerank_candidates=profile.rerank_candidates,
            rerank_top_n=profile.rerank_top_n,
            reranker_batch_size=profile.reranker_batch_size,
            device=profile.device,
        )
        self._operations = RagWarmupManager(lambda: self)
        self._knowledge_base = ChromaKnowledgeBaseAdapter(chroma_dir=chroma_dir)

    def create_index_changed_handler(self) -> "LocalRagService":
        """向建库模块公开唯一的索引变更处理器。"""
        return self

    async def notify_index_changed(self) -> None:
        await self.invalidate()

    async def start_warmup(self, force: bool = False) -> dict[str, Any]:
        return self._operations.start(force=force)

    async def get_warmup_status(self) -> dict[str, Any]:
        return self._operations.get_status()

    async def ensure_ready(
        self,
        wait_seconds: int | float = 20,
    ) -> dict[str, Any]:
        return await self._operations.ensure_ready(wait_seconds)

    async def health(self) -> RagHealthStatus:
        snapshot = self._operations.get_status()
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

    async def invalidate(self, knowledge_base_id: str = "default") -> None:
        if knowledge_base_id not in ("", "default"):
            raise ValueError("当前缓存实现仅支持 default 知识库全局失效")
        await asyncio.to_thread(self._bm25_retriever.invalidate)
        await self._query_cache.clear()

    async def read_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredChunk, ...]:
        chunks = await self._chunk_corpus.list_chunks(
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
        raise RuntimeError(
            "事件循环已运行，请使用 await LocalRagService.read_chunks(...)。"
        )

    async def close(self) -> None:
        """停止 RAG 后台任务并释放模块独占资源。"""
        resources = (
            ("operations", self._operations),
            ("bm25", self._bm25_retriever),
            ("vector", self._vector_retriever),
            ("reranker", self._reranker),
            ("redis", self._redis_resources),
        )
        for name, resource in resources:
            close = getattr(resource, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception:
                logger.exception("[RAG] 关闭资源失败 resource=%s", name)


__all__ = [
    "LocalRagService",
    "RagWarmupManager",
    "WarmupStatusSnapshot",
    "read_chunks",
    "read_chunks_sync",
]
