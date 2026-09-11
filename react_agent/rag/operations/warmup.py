"""Instance-scoped RAG warmup state machine."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from react_agent.rag.query import RetrievalService


logger = logging.getLogger(__name__)
WarmupState = Literal["not_started", "running", "done", "error", "cancelled"]


class WarmupStatusSnapshot(TypedDict):
    state: WarmupState
    started_at: float | None
    finished_at: float | None
    timings: dict[str, float]
    error: str | None


def _initial_status() -> WarmupStatusSnapshot:
    return {
        "state": "not_started",
        "started_at": None,
        "finished_at": None,
        "timings": {},
        "error": None,
    }


class RagWarmupManager:
    """Own warmup order, idempotency, failure state and task lifecycle."""

    def __init__(
        self,
        retrieval_service_provider: Callable[[], RetrievalService],
    ) -> None:
        self._retrieval_service_provider = retrieval_service_provider
        self._task: asyncio.Task | None = None
        self._status = _initial_status()

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
        """Cancel and await this manager's background task."""
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
        self._status = _initial_status()


__all__ = ["RagWarmupManager", "WarmupStatusSnapshot"]
