"""通过 HTTP 使用独立 Knowledge Service 的客户端 Runtime。"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from knowledge.contracts import (
    ChunkMetadata,
    EvaluationCandidate,
    EvaluationRetrievalResult,
    IngestionReport,
    RagHealthStatus,
    RagValidationError,
    RetrievedChunk,
    RetrievalResult,
    StoredChunk,
    WarmupStatus,
)


logger = logging.getLogger(__name__)


class KnowledgeServiceClient:
    """Knowledge Service 的异步协议客户端，同时满足查询服务接口。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout: float = 150.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        normalized = base_url.strip().rstrip("/")
        if not normalized:
            raise ValueError("Knowledge Service base_url 不能为空")
        headers = {}
        if api_key:
            headers["X-Knowledge-Service-Key"] = api_key
        self._base_url = normalized
        self._headers = headers
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=normalized,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    def _request_sync(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            with httpx.Client(
                base_url=self._base_url,
                headers=self._headers,
                timeout=httpx.Timeout(self._timeout),
            ) as client:
                response = client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ConnectionError(f"Knowledge Service 请求失败: {exc}") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = response.text
            message = (
                f"Knowledge Service 返回 HTTP {response.status_code}: "
                f"{detail or response.reason_phrase}"
            )
            if response.status_code in {400, 422}:
                raise RagValidationError(message)
            raise RuntimeError(message)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Knowledge Service 返回了非对象 JSON")
        return payload

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ConnectionError(f"Knowledge Service 请求失败: {exc}") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = response.text
            message = (
                f"Knowledge Service 返回 HTTP {response.status_code}: "
                f"{detail or response.reason_phrase}"
            )
            if response.status_code in {400, 422}:
                raise RagValidationError(message)
            raise RuntimeError(message)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Knowledge Service 返回了非对象 JSON")
        return payload

    async def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        use_query_cache: bool = True,
        retrieval_mode: str = "hybrid",
    ) -> RetrievalResult:
        payload = await self._request(
            "POST",
            "/api/v1/retrieval/search",
            json={
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "use_query_cache": use_query_cache,
                "retrieval_mode": retrieval_mode,
            },
        )
        chunks = tuple(
            RetrievedChunk(
                content=str(item.get("content") or ""),
                source_file=str(item.get("source_file") or ""),
                source_page=item.get("source_page"),
                chunk_id=str(item.get("chunk_id") or ""),
                score=item.get("score"),
                doc_type=item.get("doc_type"),
                industry=item.get("industry"),
            )
            for item in payload.get("chunks") or []
        )
        return RetrievalResult(
            query=str(payload.get("query") or query),
            chunks=chunks,
            stage=str(payload.get("stage") or "remote"),
            cache_hit=bool(payload.get("cache_hit")),
            candidates_count=int(payload.get("candidates_count") or 0),
            reranked_count=int(payload.get("reranked_count") or 0),
            top_score=float(payload.get("top_score") or 0.0),
            timings={
                str(key): float(value)
                for key, value in dict(payload.get("timings") or {}).items()
            },
        )

    async def evaluation_search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        retrieval_mode: str = "hybrid",
        use_query_cache: bool = False,
    ) -> EvaluationRetrievalResult:
        """调用 Knowledge Service 专用评测管道。"""
        if use_query_cache:
            raise RagValidationError("评测检索管道不允许使用语义查询缓存")
        payload = await self._request(
            "POST",
            "/api/v1/evaluation/retrieval/trace",
            json={
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "retrieval_mode": retrieval_mode,
            },
        )
        chunks = tuple(
            RetrievedChunk(
                content=str(item.get("content") or ""),
                source_file=str(item.get("source_file") or ""),
                source_page=item.get("source_page"),
                chunk_id=str(item.get("chunk_id") or ""),
                score=item.get("score"),
                doc_type=item.get("doc_type"),
                industry=item.get("industry"),
            )
            for item in payload.get("chunks") or []
        )
        trace = dict(payload.get("trace") or {})
        raw_stages = dict(trace.get("stages") or {})
        stages = {
            str(name): tuple(
                EvaluationCandidate(
                    rank=int(item.get("rank") or 0),
                    chunk_id=str(item.get("chunk_id") or ""),
                    source_file=str(item.get("source_file") or ""),
                    source_page=item.get("source_page"),
                    content_chars=int(item.get("content_chars") or 0),
                    doc_type=item.get("doc_type"),
                    industry=item.get("industry"),
                )
                for item in items or []
            )
            for name, items in raw_stages.items()
        }
        return EvaluationRetrievalResult(
            query=str(payload.get("query") or query),
            retrieval_mode=str(payload.get("retrieval_mode") or retrieval_mode),
            chunks=chunks,
            stages=stages,
            timings={
                str(key): float(value)
                for key, value in dict(trace.get("timings") or {}).items()
            },
            configuration=dict(trace.get("configuration") or {}),
            degraded_sources=tuple(
                str(value) for value in trace.get("degraded_sources") or []
            ),
        )

    async def request_warmup(
        self,
        *,
        wait_seconds: int | float = 20,
        force: bool = False,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/runtime/warmup",
            json={"wait_seconds": wait_seconds, "force": force},
        )

    async def warmup(self) -> WarmupStatus:
        """满足 RetrievalService 预热契约，供评测等现有调用方复用。"""
        payload = await self.request_warmup(wait_seconds=120)
        status = dict(payload.get("warmup_status") or {})
        if not payload.get("ready"):
            raise RuntimeError(
                "Knowledge Service 未完成预热: "
                f"{status.get('error') or payload.get('stage') or 'unknown'}"
            )
        return WarmupStatus(
            ready=True,
            timings={
                str(key): float(value)
                for key, value in dict(status.get("timings") or {}).items()
            },
        )

    async def warmup_status(self) -> dict[str, Any]:
        return await self._request("GET", "/api/v1/runtime/warmup")

    async def health(self) -> RagHealthStatus:
        payload = await self._request("GET", "/api/v1/admin/health")
        return RagHealthStatus(
            ready=bool(payload.get("ready")),
            state=str(payload.get("state") or "unknown"),
            details=dict(payload.get("details") or {}),
        )

    async def invalidate(self, knowledge_base_id: str = "default") -> None:
        await self._request(
            "POST",
            "/api/v1/admin/cache/invalidate",
            json={"knowledge_base_id": knowledge_base_id},
        )

    async def ingest(self, relative_path: str) -> IngestionReport:
        payload = await self._request(
            "POST",
            "/api/v1/ingestions",
            json={"relative_path": relative_path},
        )
        return self._ingestion_report(payload, relative_path)

    def ingest_sync(self, relative_path: str) -> IngestionReport:
        payload = self._request_sync(
            "POST",
            "/api/v1/ingestions",
            json={"relative_path": relative_path},
        )
        return self._ingestion_report(payload, relative_path)

    @staticmethod
    def _ingestion_report(
        payload: dict[str, Any],
        relative_path: str,
    ) -> IngestionReport:
        known = {
            "data_dir",
            "status",
            "files_discovered",
            "files_selected",
            "files_processed",
            "files_skipped",
            "chunks_written",
            "cache_invalidated",
            "details",
        }
        details = dict(payload.get("details") or {})
        details.update({key: value for key, value in payload.items() if key not in known})
        return IngestionReport(
            data_dir=str(payload.get("data_dir") or relative_path),
            status=str(payload.get("status") or "unknown"),
            files_discovered=int(payload.get("files_discovered") or 0),
            files_selected=int(payload.get("files_selected") or 0),
            files_processed=int(payload.get("files_processed") or 0),
            files_skipped=int(payload.get("files_skipped") or 0),
            chunks_written=int(payload.get("chunks_written") or 0),
            cache_invalidated=bool(payload.get("cache_invalidated")),
            details=details,
        )

    async def list_chunks_page(
        self,
        *,
        source_file: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if source_file:
            params["source_file"] = source_file
        return await self._request(
            "GET",
            "/api/v1/admin/chunks",
            params=params,
        )

    def list_chunks_page_sync(
        self,
        *,
        source_file: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if source_file:
            params["source_file"] = source_file
        return self._request_sync(
            "GET",
            "/api/v1/admin/chunks",
            params=params,
        )

    async def close(self) -> None:
        await self._client.aclose()


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
        warmup_status = payload.get("warmup_status")
        if isinstance(warmup_status, dict):
            self._last_status = {
                "task_state": (
                    "running" if warmup_status.get("state") == "running" else "done"
                ),
                "warmup_status": dict(warmup_status),
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
            payload = await self._client.list_chunks_page(
                source_file=source_file,
                offset=current,
                limit=page_size,
            )
            items = list(payload.get("items") or [])
            for item in items:
                chunks.append(
                    StoredChunk(
                        chunk_id=str(item.get("chunk_id") or ""),
                        content=str(item.get("content") or ""),
                        metadata=ChunkMetadata.from_mapping(item.get("metadata")),
                    )
                )
            if limit is not None and len(chunks) >= limit:
                return tuple(chunks[:limit])
            if not payload.get("has_more") or not items:
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
            payload = self._client.list_chunks_page_sync(
                source_file=source_file,
                offset=current,
                limit=page_size,
            )
            items = list(payload.get("items") or [])
            chunks.extend(
                StoredChunk(
                    chunk_id=str(item.get("chunk_id") or ""),
                    content=str(item.get("content") or ""),
                    metadata=ChunkMetadata.from_mapping(item.get("metadata")),
                )
                for item in items
            )
            if limit is not None and len(chunks) >= limit:
                return tuple(chunks[:limit])
            if not payload.get("has_more") or not items:
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
    "RemoteIngestionService",
    "RemoteIngestionRuntime",
    "RemoteEvaluationRetrievalService",
    "RemoteRagAdminService",
    "RemoteRagOperations",
    "RemoteRagRuntime",
]
