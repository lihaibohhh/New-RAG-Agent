"""Knowledge Service 的底层 HTTP 协议客户端。"""
from __future__ import annotations

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


class HttpRagService:
    """RAG 服务的 HTTP 实现；调用方无需感知 Remote Runtime。"""

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

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        return await self._client.search(query, **kwargs)

    async def evaluation_search(
        self,
        query: str,
        **kwargs: Any,
    ) -> EvaluationRetrievalResult:
        return await self._client.evaluation_search(query, **kwargs)

    async def start_warmup(self, force: bool = False) -> dict[str, Any]:
        return await self._client.request_warmup(wait_seconds=0, force=force)

    async def get_warmup_status(self) -> dict[str, Any]:
        return await self._client.warmup_status()

    async def ensure_ready(
        self,
        wait_seconds: int | float = 20,
    ) -> dict[str, Any]:
        return await self._client.request_warmup(wait_seconds=wait_seconds)

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

    async def close(self) -> None:
        await self._client.close()


class HttpIngestionService:
    """建库服务的 HTTP 实现。"""

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

    async def ingest(self, path: str, /) -> IngestionReport:
        return await self._client.ingest(path)

    def ingest_sync(self, path: str, /) -> IngestionReport:
        return self._client.ingest_sync(path)

    async def close(self) -> None:
        await self._client.close()


__all__ = ["HttpIngestionService", "HttpRagService", "KnowledgeServiceClient"]
