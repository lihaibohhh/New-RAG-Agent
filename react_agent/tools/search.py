from __future__ import annotations
import time
import asyncio
import logging
from collections.abc import Callable
from typing import Any
from langchain_core.tools import tool
from react_agent.tooling.results import tool_error as _err
from react_agent.tooling.results import tool_success as _ok
from react_agent.tooling.retry import with_retry

logger = logging.getLogger(__name__)


def _trim_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _normalize_search_results(
    raw: Any,
    *,
    max_items: int = 5,
    max_chars_per_item: int = 800,
) -> Any:
    """把 Tavily 响应压缩为 Agent 可消费的搜索结果。"""
    if raw is None:
        return None
    if isinstance(raw, dict) and isinstance(raw.get("results"), list):
        results = []
        for item in raw["results"][:max_items]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": _trim_text(str(item.get("title", "")), 200),
                    "url": str(item.get("url", "")),
                    "content": _trim_text(
                        str(item.get("content", "")),
                        max_chars_per_item,
                    ),
                    "score": item.get("score"),
                    "published_date": (
                        item.get("published_date") or item.get("published_time")
                    ),
                }
            )
        return {
            "results": results,
            "answer": _trim_text(str(raw.get("answer", "")), 1200),
        }
    if isinstance(raw, str):
        return _trim_text(raw, 3000)
    return raw


# ================================================================================
# 工具 1：Web 搜索（Tavily API）
# ================================================================================
# 安装依赖：pip install tavily-python
# 配置环境变量：
#   - TAVILY_API_KEY：Tavily API 密钥（必须）
#   - MAX_SEARCH_RESULTS：最多返回多少条搜索结果（可选，默认 5）
_DESCRIPTION = (
        "【触发条件】以下情况调用本工具：\n"
        "  - 需要互联网上的最新动态、实时新闻、近期公告\n"
        "  - 需要验证公开事实（公司官网信息、开源版本号、政策发布日期等）\n"
        "  - query_internal_knowledge 已返回 has_relevant_content=False，"
        "且用户问题属于可公开查询的宏观/行业信息\n"
        "  示例 query：'2024年新能源补贴政策最新消息'、'比亚迪最新季报发布时间'\n\n"
        "【不触发条件】以下情况禁止调用本工具：\n"
        "  - 用户要求查询内部知识库中的具体财务数值或研报原文\n"
        "  - query_internal_knowledge 尚未调用（不得跳过直接搜索）\n"
        "  - 用户问题涉及私有文档中的具体数据，即使知识库未命中也不得用本工具补充\n\n"
        "【输入】简洁的搜索关键词或短句，不超过300字符\n"
        "  好的输入：'比亚迪 2024 Q3 业绩发布'\n"
        "  差的输入：'请帮我在网上搜索所有关于比亚迪的信息'\n\n"
        "【输出关键字段】\n"
        "  data.results[].title     — 结果标题\n"
        "  data.results[].url       — 来源链接，回答时应标注\n"
        "  data.results[].content   — 摘要，最多800字\n"
        "  data.results[].published_date — 发布日期，判断时效性用"
)


async def _execute_search(
    query: str,
    *,
    client_provider: Callable[[int], Any],
    max_results: int,
    api_key_configured: bool,
) -> dict[str, Any]:
    """
    Web 搜索工具 — 通过 Tavily API 搜索互联网最新信息。

    功能描述：
      - 接收用户查询，通过 Tavily API 搜索互联网
      - 裁剪搜索结果以适应上下文大小
      - 返回前几条最相关的结果（标题、链接、摘要、发布时间）

    参数：
      query (str)：搜索关键词或短句
                   - 不能为空
                   - 最多 300 字符（超出会自动截断）
                   - 可以使用自然语言问题或关键词

    返回：
      {
        “ok”: True/False,
        “data”: {
          “results”: [                    # 搜索结果列表
            {
              “title”: str,               # 搜索结果标题
              “url”: str,                 # 链接
              “content”: str,             # 摘要（最多 800 字）
              “score”: float,             # 匹配分数（可能为 None）
              “published_date”: str       # 发布日期（可能为 None）
            },
            ...
          ],
          “answer”: str                   # Tavily 生成的简洁回答（可选）
        },
        “error”: {...}                    # 失败时的错误信息
      }

    执行逻辑：
      1. 输入清理：strip() 并检查是否为空
      2. 输入截断：超过 300 字符自动截断
      3. 依赖检查：延迟 import，缺少依赖返回友好错误
      4. 密钥检查：检查 TAVILY_API_KEY 环境变量
      5. 使用 Composition Root 注入的返回数量与客户端工厂
      6. API 调用：使用 TavilySearch.ainvoke() 执行搜索
      7. 结果裁剪：通过 _shrink_search_results() 压缩到适当大小
      8. 异常处理：临时错误（超时、连接）raise 给 with_retry，永久错误返回 _err()

    错误场景：
      - BAD_INPUT：query 为空
      - MISSING_DEPENDENCY：未安装 langchain-tavily
      - MISSING_API_KEY：缺少 TAVILY_API_KEY 环境变量
      - SEARCH_FAILED：API 调用出错（非临时错误）
      - 超时/连接错误：raise，由 with_retry 自动重试 2 次

    示例：
      result = await search(“Python 3.12 最新功能”)
      # 返回关于 Python 3.12 的最新搜索结果
    """
    tool_name = "search"
    q = (query or "").strip()

    # ════════════════════ 阶段 1：输入校验 ════════════════════
    if not q:
        return _err(
            tool_name=tool_name,
            query=q,
            code="BAD_INPUT",
            message="query 不能为空，请提供要搜索的关键词。"
        )
    if len(q) > 300:
        q = q[:300]

    # ════════════════════ 阶段 2：密钥检查 ════════════════════
    # Tavily API 需要密钥认证，提前检查能更快给出友好错误
    if not api_key_configured:
        return _err(
            tool_name=tool_name,
            query=q,
            code="MISSING_API_KEY",
            message="缺少环境变量 TAVILY_API_KEY，搜索工具不可用。请先设置该 Key。",
        )

    # ════════════════════ 阶段 3：执行搜索 ════════════════════
    try:
        wrapped = client_provider(max_results)

        # 调用搜索 API（Tavily 支持两种调用签名）
        logger.info("[search] 发起 Tavily 请求 | query=%r | max_results=%s", q, max_results)
        _t0 = time.perf_counter()
        try:
            raw = await wrapped.ainvoke({"query": q})
        except Exception:
            # 如果字典形式失败，尝试直接传字符串
            raw = await wrapped.ainvoke(q)
        _elapsed = time.perf_counter() - _t0
        _hit_count = len(raw.get("results", [])) if isinstance(raw, dict) else -1
        logger.info("[search] Tavily 返回 | query=%r | 耗时=%.2fs | 命中=%d 条", q, _elapsed, _hit_count)

        # 裁剪结果以适应上下文
        data = _normalize_search_results(raw, max_items=min(5, max_results))
        return _ok(
            tool_name=tool_name,
            query=q,
            data=data,
            meta={"max_results": max_results},
        )

    # ════════════════════ 阶段 4：异常处理 ════════════════════
    except (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError):
        # 临时错误：raise 给 with_retry 装饰器，自动重试
        # 不能在这里 try-except 吞掉，否则 with_retry 无法捕获
        raise
    except ImportError:
        return _err(
            tool_name=tool_name,
            query=q,
            code="MISSING_DEPENDENCY",
            message="未安装依赖 langchain-tavily。请执行：pip install langchain-tavily",
        )
    except Exception as e:
        # 永久错误（API 返回 4xx、解析失败等）：返回失败结果，不重试
        return _err(
            tool_name=tool_name,
            query=q,
            code="SEARCH_FAILED",
            message=f"搜索工具调用失败：{type(e).__name__}: {e}",
        )


def create_search_tool(
    *,
    client_provider: Callable[[int], Any],
    max_results: int,
    api_key_configured: bool,
    max_retries: int,
    timeout: float,
):
    """Create a search adapter from dependencies selected by Runtime."""
    safe_max_results = max(1, min(int(max_results), 20))

    @tool(description=_DESCRIPTION)
    @with_retry(
        tool_name="search",
        max_retries=max_retries,
        timeout=timeout,
    )
    async def search(query: str) -> dict[str, Any]:
        return await _execute_search(
            query,
            client_provider=client_provider,
            max_results=safe_max_results,
            api_key_configured=api_key_configured,
        )

    return search


__all__ = ["create_search_tool"]
