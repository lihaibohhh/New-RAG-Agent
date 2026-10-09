"""唯一持有本地 Chroma 的 Knowledge Service 应用。"""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status

from knowledge.transport.schemas import (
    EvaluationSearchRequest,
    IngestionRequest,
    InvalidateRequest,
    SearchRequest,
    WarmupRequest,
)
from knowledge.services import IngestionService, RagService
from knowledge.settings import KnowledgeServiceSettings, KnowledgeSettings
from knowledge.contracts import KnowledgeValidationError
from knowledge.transport.codecs import (
    evaluation_result_to_payload,
    health_status_to_payload,
    ingestion_report_to_payload,
    retrieval_result_to_payload,
    stored_chunk_page_to_payload,
    warmup_response_to_payload,
)
from knowledge.server.runtime import create_knowledge_services_from_env


logger = logging.getLogger(__name__)


class KnowledgeServicesPort(Protocol):
    rag: RagService
    ingestion: IngestionService

    async def close(self) -> None: ...


RuntimeFactory = Callable[[], KnowledgeServicesPort]


def _resolve_ingestion_path(root: Path, relative_path: str) -> Path:
    requested = Path(relative_path)
    if requested.is_absolute() or PureWindowsPath(relative_path).is_absolute():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="relative_path 必须是服务端语料根目录下的相对路径",
        )
    resolved_root = root.resolve()
    resolved = (resolved_root / requested).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="relative_path 不能越过服务端语料根目录",
        ) from exc
    return resolved


def create_app(
    *,
    runtime_factory: RuntimeFactory = create_knowledge_services_from_env,
    service_settings: KnowledgeServiceSettings | None = None,
) -> FastAPI:
    """创建可注入测试 Runtime 的 Knowledge Service。"""
    configured = service_settings or KnowledgeSettings.from_env().service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory()
        app.state.knowledge_services = runtime
        app.state.rag_service = runtime.rag
        app.state.ingestion_service = runtime.ingestion
        app.state.knowledge_settings = configured
        if configured.require_api_key and not configured.api_key:
            await runtime.close()
            raise RuntimeError(
                "KNOWLEDGE_SERVICE_REQUIRE_API_KEY=true，"
                "但 KNOWLEDGE_SERVICE_API_KEY 未配置"
            )
        if not configured.api_key:
            logger.warning(
                "[KnowledgeService] KNOWLEDGE_SERVICE_API_KEY 未配置；"
                "管理与检索接口当前未启用鉴权"
            )
        if configured.warmup_on_start:
            await runtime.rag.start_warmup(force=False)
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(
        title="Financial Knowledge Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    def rag_service(request: Request) -> RagService:
        return request.app.state.rag_service

    def ingestion_service(request: Request) -> IngestionService:
        return request.app.state.ingestion_service

    def require_api_key(
        request: Request,
        supplied: str | None = Header(
            default=None,
            alias="X-Knowledge-Service-Key",
        ),
    ) -> None:
        expected = request.app.state.knowledge_settings.api_key
        if expected and not hmac.compare_digest(supplied or "", expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Knowledge Service API key 无效",
            )

    protected = [Depends(require_api_key)]

    @app.exception_handler(KnowledgeValidationError)
    async def rag_validation_error(
        _request: Request,
        exc: KnowledgeValidationError,
    ) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/v1/health/live", tags=["system"])
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/health/ready", tags=["system"])
    async def ready(request: Request) -> dict[str, Any]:
        try:
            health = await rag_service(request).health()
        except Exception as exc:
            logger.exception("[KnowledgeService] readiness 检查失败")
            return {
                "ready": False,
                "state": "knowledge_base_error",
                "error_type": type(exc).__name__,
            }
        knowledge = dict(health.details.get("knowledge_base") or {})
        return {
            "ready": health.ready,
            "state": health.state,
            "chunk_count": knowledge.get("chunk_count"),
        }

    @app.post(
        "/api/v1/retrieval/search",
        dependencies=protected,
        tags=["retrieval"],
    )
    async def search(body: SearchRequest, request: Request) -> dict[str, Any]:
        result = await rag_service(request).search(
            body.query,
            top_k=body.top_k,
            filters=body.filters,
            use_query_cache=body.use_query_cache,
            retrieval_mode=body.retrieval_mode,
        )
        return retrieval_result_to_payload(result)

    @app.post(
        "/api/v1/evaluation/retrieval/trace",
        dependencies=protected,
        tags=["evaluation"],
    )
    async def evaluation_retrieval_trace(
        body: EvaluationSearchRequest,
        request: Request,
    ) -> dict[str, Any]:
        if not request.app.state.knowledge_settings.evaluation_api_enabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation retrieval API 未启用",
            )
        result = await rag_service(request).evaluation_search(
            body.query,
            top_k=body.top_k,
            filters=body.filters,
            retrieval_mode=body.retrieval_mode,
        )
        return evaluation_result_to_payload(result)

    @app.get(
        "/api/v1/runtime/warmup",
        dependencies=protected,
        tags=["operations"],
    )
    async def warmup_status(request: Request) -> dict[str, Any]:
        return warmup_response_to_payload(
            await rag_service(request).get_warmup_status()
        )

    @app.post(
        "/api/v1/runtime/warmup",
        dependencies=protected,
        tags=["operations"],
    )
    async def warmup(body: WarmupRequest, request: Request) -> dict[str, Any]:
        if body.force:
            await rag_service(request).start_warmup(force=True)
        return warmup_response_to_payload(
            await rag_service(request).ensure_ready(body.wait_seconds)
        )

    @app.get(
        "/api/v1/admin/health",
        dependencies=protected,
        tags=["admin"],
    )
    async def admin_health(request: Request) -> dict[str, Any]:
        health = await rag_service(request).health()
        return health_status_to_payload(health)

    @app.post(
        "/api/v1/admin/cache/invalidate",
        dependencies=protected,
        tags=["admin"],
    )
    async def invalidate(
        body: InvalidateRequest,
        request: Request,
    ) -> dict[str, Any]:
        await rag_service(request).invalidate(body.knowledge_base_id)
        return {"status": "invalidated", "knowledge_base_id": body.knowledge_base_id}

    @app.get(
        "/api/v1/admin/chunks",
        dependencies=protected,
        tags=["admin"],
    )
    async def chunks(
        request: Request,
        source_file: str | None = None,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=1000),
    ) -> dict[str, Any]:
        items = await rag_service(request).read_chunks(
            source_file=source_file,
            offset=offset,
            limit=limit,
        )
        return stored_chunk_page_to_payload(
            items,
            offset=offset,
            limit=limit,
            has_more=len(items) == limit,
        )

    @app.post(
        "/api/v1/ingestions",
        dependencies=protected,
        tags=["ingestion"],
    )
    async def ingest(body: IngestionRequest, request: Request) -> dict[str, Any]:
        root = request.app.state.knowledge_settings.ingestion_root
        data_path = _resolve_ingestion_path(root, body.relative_path)
        report = await ingestion_service(request).ingest(str(data_path))
        return ingestion_report_to_payload(report)

    return app


__all__ = ["create_app"]
