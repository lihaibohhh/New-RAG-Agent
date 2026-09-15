"""LangChain Tool 到统一 RAG 查询服务的薄适配层。"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

from react_agent.rag.contracts import RagValidationError, RetrievalResult
from react_agent.tooling.results import tool_error as _err
from react_agent.tooling.results import tool_success as _ok
from react_agent.tooling.retry import with_retry


TOOL_NAME = "query_internal_knowledge"

_DESCRIPTION = (
    "【触发条件】用户询问具体公司财务数据、行业研报观点、券商评级、目标价、"
    "盈利预测、已入库政策条款或研报图表数据时，必须优先调用本工具。\n"
    "【不触发条件】通用概念解释或用户明确要求互联网最新信息时不使用。\n"
    "【输入】使用公司名 + 指标 + 报告期等精炼金融关键词。\n"
    "【输出】data.results 返回正文、来源文档、页码和 chunk_id；"
    "has_relevant_content=False 时禁止根据知识库编造答案。"
)


def create_rag_tool(
    *,
    retrieval_service_provider: Callable[[], Any],
    max_retries: int,
    timeout: float,
):
    """创建只依赖已注入 RetrievalService 提供者的 Agent Tool。

    Adapter 不再知道 RAG Runtime 的全局容器；服务选择、重试与超时
    参数由顶层 Composition Root 在组装时确定。
    """

    @tool(description=_DESCRIPTION)
    @with_retry(
        tool_name=TOOL_NAME,
        max_retries=max_retries,
        timeout=timeout,
        # 模型计算超时后底层线程不能被安全终止，禁止立即重复提交计算。
        retry_on=(ConnectionError, OSError),
        retry_timeouts=False,
    )
    async def query_internal_knowledge(query: str) -> dict:
        """调用已注入的 RetrievalService，并转换为 Tool 返回结构。"""
        q = (query or "").strip()
        if not q:
            return _err(
                tool_name=TOOL_NAME,
                query=q,
                code="BAD_INPUT",
                message="检索词不能为空",
            )

        try:
            result = await retrieval_service_provider().search(q, top_k=3)
            return _to_tool_payload(result)
        except RagValidationError as exc:
            return _err(
                tool_name=TOOL_NAME,
                query=q,
                code="BAD_INPUT",
                message=str(exc),
            )
        except (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError):
            raise
        except Exception as exc:
            return _err(
                tool_name=TOOL_NAME,
                query=q,
                code="RAG_SEARCH_FAILED",
                message=f"知识库检索失败: {type(exc).__name__}: {exc}",
            )

    return query_internal_knowledge


def _to_tool_payload(result: RetrievalResult) -> dict:
    results: list[dict] = []
    for chunk in result.chunks:
        item = chunk.to_dict()
        page = chunk.source_page
        if page is not None and page >= 1:
            prefix = f"[来源：{chunk.source_file}  第 {page} 页]"
        else:
            prefix = f"[来源：{chunk.source_file}]"
        item["content"] = f"{prefix}\n{chunk.content}"
        results.append(item)

    has_content = result.has_relevant_content
    return _ok(
        tool_name=TOOL_NAME,
        query=result.query,
        data={
            "results": results,
            "has_relevant_content": has_content,
        },
        meta={
            "retrieved_count": len(results),
            "candidates_count": result.candidates_count,
            "reranked_count": result.reranked_count,
            "stage": result.stage,
            "cache_hit": result.cache_hit,
            "top_score": round(result.top_score, 4),
            "has_relevant_content": has_content,
            "timings": result.timings,
        },
    )


__all__ = ["create_rag_tool"]
