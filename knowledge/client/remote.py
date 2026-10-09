"""Knowledge Service 的远程能力适配器与 Runtime 视图。"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from knowledge.client.http import KnowledgeServiceClient
from knowledge.contracts import (
    EvaluationRetrievalResult,
    IngestionReport,
    RagHealthStatus,
    StoredChunk,
)
from knowledge.transport.codecs import warmup_response_from_payload


logger = logging.getLogger(__name__)


class RemoteRagOperations:
    """把本地预热管理接口映射到服务端的单例预热状态。"""

    def __init__(self, client: KnowledgeServiceClient) -> None:
        self._client = client
        self._task: asyncio.Task[None] | None = None
        self._last_status: dict[str, Any] = {
            "task_state": "none",
            "warmup_status": {
                "state": "not_started",
                "started_at": None,
                "finished_at": None,
                "timings": {},
                "error": None,
            },
        }

    def _record(self, payload: dict[str, Any]) -> None:
        response = warmup_response_from_payload(payload)
        warmup_status = response.warmup_status.model_dump(mode="json")
        self._last_status = {
            "task_state": response.task_state
            or ("running" if response.warmup_status.state == "running" else "done"),
            "warmup_status": warmup_status,
        }

    async def _run(self, force: bool) -> None:
        try:
            self._record(
                await self._client.request_warmup(wait_seconds=120, force=force)
            )
        except Exception as exc:
            logger.exception("[RAG] 远程 Knowledge Service 预热失败")
            self._last_status = {
                "task_state": "done",
                "warmup_status": {
                    "state": "error",
                    "started_at": None,
                    "finished_at": time.time(),
                    "timings": {},
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }

    def start(self, force: bool = False) -> dict[str, Any]:
        if self._task is not None and not self._task.done():
            return {"stage": "already_running", **self.get_status()}
        self._task = asyncio.create_task(
            self._run(force),
            name="remote_rag_service_warmup",
        )
        return {"stage": "started", "task_state": "running"}

    def get_status(self) -> dict[str, Any]:
        return {
            "task_state": (
                "running"
                if self._task is not None and not self._task.done()
                else self._last_status.get("task_state", "none")
            ),
            "warmup_status": dict(self._last_status.get("warmup_status") or {}),
        }

    async def ensure_ready(
        self,
        wait_seconds: int | float = 20,
    ) -> dict[str, Any]:
        payload = await self._client.request_warmup(wait_seconds=wait_seconds)
        self._record(payload)
        return payload

    async def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None


class RemoteRagAdminService:
    def __init__(self, client: KnowledgeServiceClient) -> None:
        self._client = client

    async def health(self) -> RagHealthStatus:
        return await self._client.health()

    async def invalidate(self, knowledge_base_id: str = "default") -> None:
        await self._client.invalidate(knowledge_base_id)

    async def read_chunks(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredChunk, ...]:
        page_size = min(limit or 1000, 1000)
        current = max(0, offset)
        chunks: list[StoredChunk] = []
        while True:
            page = await self._client.list_chunks_page(
                source_file=source_file,
                offset=current,
                limit=page_size,
            )
            items = list(page.items)
            chunks.extend(items)
            if limit is not None and len(chunks) >= limit:
                return tuple(chunks[:limit])
            if not page.has_more or not items:
                return tuple(chunks)
            current += len(items)

    def read_chunks_sync(
        self,
        *,
        source_file: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[StoredChunk, ...]:
        page_size = min(limit or 1000, 1000)
        current = max(0, offset)
        chunks: list[StoredChunk] = []
        while True:
            page = self._client.list_chunks_page_sync(
                source_file=source_file,
                offset=current,
                limit=page_size,
            )
            items = list(page.items)
            chunks.extend(items)
            if limit is not None and len(chunks) >= limit:
                return tuple(chunks[:limit])
            if not page.has_more or not items:
                return tuple(chunks)
            current += len(items)


class RemoteIngestionService:
    def __init__(self, client: KnowledgeServiceClient) -> None:
        self._client = client

    async def ingest(self, relative_path: str) -> IngestionReport:
        return await self._client.ingest(relative_path)

    def ingest_sync(self, relative_path: str) -> IngestionReport:
        return self._client.ingest_sync(relative_path)


class RemoteEvaluationRetrievalService:
    """远程评测检索用例，与普通查询客户端分开暴露。"""

    def __init__(self, client: KnowledgeServiceClient) -> None:
        self._client = client

    async def search(self, query: str, **kwargs: Any) -> EvaluationRetrievalResult:
        return await self._client.evaluation_search(query, **kwargs)


class RemoteRagRuntime:
    """远程在线 RAG 能力；不暴露建库接口。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout: float = 150.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = KnowledgeServiceClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            transport=transport,
        )
        self.operations = RemoteRagOperations(self._client)
        self._admin = RemoteRagAdminService(self._client)
        self._evaluation = RemoteEvaluationRetrievalService(self._client)

    def get_retrieval_service(self) -> KnowledgeServiceClient:
        return self._client

    def get_evaluation_retrieval_service(
        self,
    ) -> RemoteEvaluationRetrievalService:
        return self._evaluation

    def get_admin_service(self) -> RemoteRagAdminService:
        return self._admin

    async def close(self) -> None:
        await self.operations.close()
        await self._client.close()


class RemoteIngestionRuntime:
    """远程建库能力；不暴露查询、评测或 RAG 管理接口。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout: float = 150.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = KnowledgeServiceClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            transport=transport,
        )
        self._ingestion = RemoteIngestionService(self._client)

    def get_ingestion_service(self) -> RemoteIngestionService:
        return self._ingestion

    async def close(self) -> None:
        await self._client.close()


__all__ = [
    "KnowledgeServiceClient",
    "RemoteEvaluationRetrievalService",
    "RemoteIngestionRuntime",
    "RemoteIngestionService",
    "RemoteRagAdminService",
    "RemoteRagOperations",
    "RemoteRagRuntime",
]
