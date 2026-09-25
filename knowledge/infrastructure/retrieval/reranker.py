"""Cross-Encoder 精排端口实现。"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from knowledge.contracts import RagDocument


logger = logging.getLogger(__name__)
_inference_pending = 0
_inference_pending_lock = threading.Lock()


def _score_with_gate(
    reranker: Any,
    pairs: list[tuple[str, str]],
    gate: threading.BoundedSemaphore,
    batch_size: int,
) -> tuple[list[float], float, float, int, int]:
    """在指定并发闸门内执行同步评分。"""
    global _inference_pending
    queued_at = time.perf_counter()
    with _inference_pending_lock:
        _inference_pending += 1
        pending = _inference_pending
    try:
        gate.acquire()
        queue_wait = time.perf_counter() - queued_at
        inference_started = time.perf_counter()
        scores: list[float] = []
        batch_count = 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            scores.extend(float(score) for score in reranker.score(batch))
            batch_count += 1
        inference_elapsed = time.perf_counter() - inference_started
        return scores, queue_wait, inference_elapsed, pending, batch_count
    finally:
        gate.release()
        with _inference_pending_lock:
            _inference_pending -= 1


class RerankerProviderAdapter:
    """设备和并发策略由 Composition Root 显式注入的 BGE 精排器。"""

    def __init__(
        self,
        *,
        device: str,
        inference_concurrency: int,
        batch_size: int,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        threshold: float = 0.1,
        debug: bool = False,
    ) -> None:
        if not device:
            raise ValueError("Reranker device 不能为空")
        if inference_concurrency < 1:
            raise ValueError("Reranker inference_concurrency 必须大于 0")
        if batch_size < 1:
            raise ValueError("Reranker batch_size 必须大于 0")
        self._device = device
        self._model_name = model_name
        self._threshold = float(threshold)
        self._debug = bool(debug)
        self._batch_size = batch_size
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._gate = threading.BoundedSemaphore(inference_concurrency)

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from langchain_community.cross_encoders import (
                    HuggingFaceCrossEncoder,
                )
            except ImportError as exc:
                raise ImportError(
                    "缺少依赖。请安装 langchain-community 和 sentence-transformers"
                ) from exc

            logger.info(
                "[RAG] Reranker 初始化 model=%s device=%s",
                self._model_name,
                self._device,
            )
            self._model = HuggingFaceCrossEncoder(
                model_name=self._model_name,
                model_kwargs={"device": self._device},
            )
            return self._model

    async def warmup(self) -> None:
        model = await asyncio.to_thread(self._get_model)
        _, queue_wait, inference_elapsed, pending, _batch_count = (
            await asyncio.to_thread(
                _score_with_gate,
                model,
                [("知识库检索预热", "这是用于初始化精排模型计算图的预热文本。")],
                self._gate,
                self._batch_size,
            )
        )
        logger.info(
            "[RAG] Reranker warmup queue_wait=%.3fs inference=%.3fs pending=%s",
            queue_wait,
            inference_elapsed,
            pending,
        )

    async def rerank(
        self,
        query: str,
        documents: list[RagDocument],
        *,
        top_n: int,
    ) -> tuple[list[RagDocument], float]:
        if not documents:
            return [], 0.0

        try:
            model = await asyncio.to_thread(self._get_model)
            pairs = [(query, document.content) for document in documents]
            scores, queue_wait, inference_elapsed, pending, batch_count = (
                await asyncio.to_thread(
                    _score_with_gate,
                    model,
                    pairs,
                    self._gate,
                    self._batch_size,
                )
            )
            logger.info(
                "[RAG] Reranker scored pairs=%s batches=%s batch_size=%s "
                "queue_wait=%.3fs inference=%.3fs pending=%s",
                len(pairs),
                batch_count,
                self._batch_size,
                queue_wait,
                inference_elapsed,
                pending,
            )
            scored = sorted(
                zip(scores, documents),
                key=lambda item: item[0],
                reverse=True,
            )
            threshold = self._threshold
            filtered = [item for item in scored if item[0] > threshold]
            if not filtered:
                logger.info(
                    "[RAG] Reranker 全部低于阈值 threshold=%s",
                    threshold,
                )
                return [], 0.0

            if self._debug:
                for score, document in scored:
                    logger.info(
                        "[RAG][DEBUG] score=%.3f chunk=%s retained=%s",
                        score,
                        document.metadata.chunk_id or "?",
                        score > threshold,
                    )
            return (
                [document for _, document in filtered[:top_n]],
                filtered[0][0],
            )
        except Exception as exc:
            logger.warning(
                "[RAG] Reranker 失败，降级返回 RRF Top-%s: %s: %s",
                top_n,
                type(exc).__name__,
                exc,
            )
            return documents[:top_n], 0.0

    async def close(self) -> None:
        """Release the runtime's strong reference to the model."""
        with self._model_lock:
            self._model = None


__all__ = ["RerankerProviderAdapter"]
