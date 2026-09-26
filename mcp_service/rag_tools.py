"""MCP 到统一 RAG 查询服务的薄协议适配层。"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from mcp_service.responses import clamp_int, mcp_err, mcp_ok
from knowledge.contracts import KnowledgeValidationError, RetrievalResult
from knowledge.runtime_ports import RetrievalServicePort
from mcp_service.observability import ToolCallTrace

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


logger = logging.getLogger(__name__)

QUERY_FINANCIAL_REPORTS_DESCRIPTION = (
    "在已入库金融研报私有知识库中检索证据片段。不是实时数据、联网搜索或结构化 SQL 查询工具。\n"
    "query 建议使用公司名 + 指标 + 报告期等精炼关键词。"
    "top_k 默认 3，范围 1~10。首次查询会自动检查 RAG 查询服务预热状态。"
    "返回正文以及来源文档、页码、chunk_id、doc_type、industry 等可用元数据。"
)

WarmupCallable = Callable[[int | float], Awaitable[dict[str, Any]]]
RetrievalServiceProvider = Callable[[], RetrievalServicePort]


def register_rag_tools(
    server: FastMCP,
    *,
    service_provider: RetrievalServiceProvider,
    warmup: WarmupCallable,
) -> None:
    @server.tool(description=QUERY_FINANCIAL_REPORTS_DESCRIPTION)
    async def query_financial_reports(
        query: str,
        top_k: int = 3,
        wait_for_ready_seconds: int = 20,
        include_meta: bool = True,
    ) -> dict[str, Any]:
        return await execute_query_financial_reports(
            query=query,
            top_k=top_k,
            wait_for_ready_seconds=wait_for_ready_seconds,
            include_meta=include_meta,
            service_provider=service_provider,
            warmup=warmup,
        )


async def execute_query_financial_reports(
    *,
    service_provider: RetrievalServiceProvider,
    warmup: WarmupCallable,
    query: str,
    top_k: int = 3,
    wait_for_ready_seconds: int = 20,
    include_meta: bool = True,
) -> dict[str, Any]:
    """Execute an MCP query through explicitly injected use cases."""
    q = (query or "").strip()
    safe_top_k = clamp_int(top_k, default=3, min_value=1, max_value=10)
    safe_wait = clamp_int(
        wait_for_ready_seconds,
        default=20,
        min_value=0,
        max_value=120,
    )
    trace = ToolCallTrace.start("query_financial_reports")

    if not q:
        meta = _meta(
            include_meta=include_meta,
            stage="validation",
            top_k=safe_top_k,
        )
        meta.update(
            trace.finish(
                stage="validation",
                status="error",
                error_type="ValidationError",
            )
        )
        return mcp_err(
            "检索词不能为空",
            error_type="ValidationError",
            meta=meta,
        )

    try:
        ready = await warmup(safe_wait)
    except Exception as exc:
        logger.exception("[RAG-MCP] 预热检查异常 query_chars=%s", len(q))
        meta = _meta(
            include_meta=include_meta,
            stage="warmup_exception",
            top_k=safe_top_k,
        )
        meta.update(
            trace.finish(
                stage="warmup_exception",
                status="error",
                error_type=type(exc).__name__,
            )
        )
        return mcp_err(
            f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
            meta=meta,
        )
    if not ready.get("ready"):
        return _warmup_not_ready_response(
            query=q,
            top_k=safe_top_k,
            include_meta=include_meta,
            ready=ready,
            trace=trace,
        )

    try:
        result = await service_provider().search(q, top_k=safe_top_k)
        meta = _result_meta(
            result,
            top_k=safe_top_k,
            include_meta=include_meta,
            warmup_ready=ready,
        )
        meta.update(trace.finish(stage=result.stage, status="ok"))
        return mcp_ok(
            data=_result_data(result),
            meta=meta,
        )
    except KnowledgeValidationError as exc:
        meta = _meta(
            include_meta=include_meta,
            stage="validation",
            top_k=safe_top_k,
        )
        meta.update(
            trace.finish(
                stage="validation",
                status="error",
                error_type="ValidationError",
            )
        )
        return mcp_err(
            str(exc),
            error_type="ValidationError",
            meta=meta,
        )
    except Exception as exc:
        logger.exception("[RAG-MCP] 检索异常 query_chars=%s", len(q))
        meta = _meta(
            include_meta=include_meta,
            stage="exception",
            top_k=safe_top_k,
        )
        meta.update(
            trace.finish(
                stage="exception",
                status="error",
                error_type=type(exc).__name__,
            )
        )
        return mcp_err(
            f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
            meta=meta,
        )


def _warmup_not_ready_response(
    *,
    query: str,
    top_k: int,
    include_meta: bool,
    ready: dict[str, Any],
    trace: ToolCallTrace,
) -> dict[str, Any]:
    stage = str(ready.get("stage") or "warming_up")
    data = {
        "query": query,
        "has_relevant_content": False,
        "results": [],
        "warmup_status": ready.get("warmup_status"),
    }
    meta = _meta(
        include_meta=include_meta,
        stage=stage,
        top_k=top_k,
        warmup_ready=ready,
    )
    if stage in {"warmup_error", "warmup_cancelled"}:
        meta.update(
            trace.finish(
                stage=stage,
                status="error",
                error_type="WarmupNotReady",
            )
        )
        return mcp_err(
            "RAG 查询服务预热未完成，请检查 warmup_status 后重试。",
            error_type="WarmupNotReady",
            data=data,
            meta=meta,
        )

    meta.update(trace.finish(stage=stage, status="not_ready"))
    data.update(
        status="warming_up",
        message="RAG 查询服务仍在预热，请稍后重试同一查询。",
        retry_after_seconds=ready.get("retry_after_seconds"),
    )
    return mcp_ok(data=data, meta=meta)


def _result_data(result: RetrievalResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "has_relevant_content": result.has_relevant_content,
        "results": result.results(),
    }


def _result_meta(
    result: RetrievalResult,
    *,
    top_k: int,
    include_meta: bool,
    warmup_ready: dict[str, Any],
) -> dict[str, Any]:
    return _meta(
        include_meta=include_meta,
        stage=result.stage,
        top_k=top_k,
        count=len(result.chunks),
        candidates=result.candidates_count,
        reranked=result.reranked_count,
        top_score=round(result.top_score, 4),
        cache_hit=result.cache_hit,
        timings=result.timings,
        warmup_ready=warmup_ready,
    )


def _meta(
    *,
    include_meta: bool,
    stage: str,
    top_k: int | None = None,
    count: int | None = None,
    candidates: int | None = None,
    reranked: int | None = None,
    top_score: float | None = None,
    cache_hit: bool | None = None,
    timings: dict[str, float] | None = None,
    warmup_ready: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {"tool": "query_financial_reports", "stage": stage}
    if not include_meta:
        return base
    optional = {
        "top_k": top_k,
        "count": count,
        "candidates": candidates,
        "reranked": reranked,
        "top_score": top_score,
        "cache_hit": cache_hit,
        "timings": timings,
    }
    base.update({key: value for key, value in optional.items() if value is not None})
    if warmup_ready is not None:
        base["warmup_stage"] = warmup_ready.get("stage")
        base["warmup_waited_seconds"] = warmup_ready.get("waited_seconds")
        if warmup_ready.get("retry_after_seconds") is not None:
            base["retry_after_seconds"] = warmup_ready.get("retry_after_seconds")
    return base


__all__ = ["execute_query_financial_reports", "register_rag_tools"]
