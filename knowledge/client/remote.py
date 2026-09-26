"""通过 HTTP 使用独立 Knowledge Service 的客户端 Runtime。"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from knowledge.contracts import (
    EvaluationRetrievalResult,
    IngestionReport,
    KnowledgeValidationError,
    RagHealthStatus,
    RetrievalResult,
    StoredChunk,
    WarmupStatus,
)
from knowledge.transport.codecs import (
    DecodedChunkPage,
    evaluation_result_from_payload,
    health_status_from_payload,
    ingestion_report_from_payload,
    retrieval_result_from_payload,
    stored_chunk_page_from_payload,
    warmup_response_from_payload,
    warmup_response_to_payload,
    warmup_status_from_payload,
)
from knowledge.transport.schemas import (
    EvaluationSearchRequest,
    IngestionRequest,
    InvalidateRequest,
    SearchRequest,
    WarmupRequest,
)


logger = logging.getLogger(__name__)
_RequestT = TypeVar("_RequestT", bound=BaseModel)


def _validated_request(
    request_type: type[_RequestT],
    **values: Any,
) -> _RequestT:
    try:
        return request_type(**values)
    except ValidationError as exc:
        raise KnowledgeValidationError(
            f"Knowledge Service 请求参数不符合 {request_type.__name__}: {exc}"
        ) from exc


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
                raise KnowledgeValidationError(message)
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
                raise KnowledgeValidationError(message)
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
        request = _validated_request(
            SearchRequest,
            query=query,
            top_k=top_k,
            filters=filters,
            use_query_cache=use_query_cache,
            retrieval_mode=retrieval_mode,
        )
        payload = await self._request(
            "POST",
            "/api/v1/retrieval/search",
            json=request.model_dump(mode="json"),
        )
        return retrieval_result_from_payload(
            payload,
            fallback_query=request.query,
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
            raise KnowledgeValidationError("评测检索管道不允许使用语义查询缓存")
        request = _validated_request(
            EvaluationSearchRequest,
            query=query,
            top_k=top_k,
            filters=filters,
            retrieval_mode=retrieval_mode,
        )
        payload = await self._request(
            "POST",
            "/api/v1/evaluation/retrieval/trace",
            json=request.model_dump(mode="json"),
        )
        return evaluation_result_from_payload(
            payload,
            fallback_query=request.query,
            fallback_retrieval_mode=request.retrieval_mode,
        )

    async def request_warmup(
        self,
        *,
        wait_seconds: int | float = 20,
        force: bool = False,
    ) -> dict[str, Any]:
        request = _validated_request(
            WarmupRequest,
            wait_seconds=wait_seconds,
            force=force,
        )
        return warmup_response_to_payload(
            await self._request(
                "POST",
                "/api/v1/runtime/warmup",
                json=request.model_dump(mode="json"),
            )
        )

    async def warmup(self) -> WarmupStatus:
        """满足 RetrievalService 预热契约，供评测等现有调用方复用。"""
        payload = await self.request_warmup(wait_seconds=120)
        return warmup_status_from_payload(payload)

    async def warmup_status(self) -> dict[str, Any]:
        return warmup_response_to_payload(
            await self._request("GET", "/api/v1/runtime/warmup")
        )

    async def health(self) -> RagHealthStatus:
        payload = await self._request("GET", "/api/v1/admin/health")
        return health_status_from_payload(payload)

    async def invalidate(self, knowledge_base_id: str = "default") -> None:
        request = _validated_request(
            InvalidateRequest,
            knowledge_base_id=knowledge_base_id,
        )
        await self._request(
            "POST",
            "/api/v1/admin/cache/invalidate",
            json=request.model_dump(mode="json"),
        )

    async def ingest(self, relative_path: str) -> IngestionReport:
        request = _validated_request(
            IngestionRequest,
            relative_path=relative_path,
        )
        payload = await self._request(
            "POST",
            "/api/v1/ingestions",
            json=request.model_dump(mode="json"),
        )
        return ingestion_report_from_payload(
            payload,
            fallback_data_dir=request.relative_path,
        )

    def ingest_sync(self, relative_path: str) -> IngestionReport:
        request = _validated_request(
            IngestionRequest,
            relative_path=relative_path,
        )
        payload = self._request_sync(
            "POST",
            "/api/v1/ingestions",
            json=request.model_dump(mode="json"),
        )
        return ingestion_report_from_payload(
            payload,
            fallback_data_dir=request.relative_path,
        )

    async def list_chunks_page(
        self,
        *,
        source_file: str | None,
        offset: int,
        limit: int,
    ) -> DecodedChunkPage:
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if source_file:
            params["source_file"] = source_file
        return stored_chunk_page_from_payload(
            await self._request(
                "GET",
                "/api/v1/admin/chunks",
                params=params,
            )
        )

    def list_chunks_page_sync(
        self,
        *,
        source_file: str | None,
        offset: int,
        limit: int,
    ) -> DecodedChunkPage:
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if source_file:
            params["source_file"] = source_file
        return stored_chunk_page_from_payload(
            self._request_sync(
                "GET",
                "/api/v1/admin/chunks",
                params=params,
            )
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
    "RemoteIngestionService",
    "RemoteIngestionRuntime",
    "RemoteEvaluationRetrievalService",
    "RemoteRagAdminService",
    "RemoteRagOperations",
    "RemoteRagRuntime",
]
