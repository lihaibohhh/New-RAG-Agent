"""工具结果裁剪、批次统计与动态执行。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from agent_tools.contracts.retrieval import (
    bound_retrieval_payload,
    decode_retrieval_outcome,
    is_retrieval_payload,
)
from agent_tools.contracts.results import decode_tool_result, tool_error
from react_agent.agent.contracts.dependencies import (
    AgentDependencies,
    ToolCapability,
)
from react_agent.agent.contracts.state import State
from react_agent.agent.policies import TerminationReason
from react_agent.agent.tool_flow.protocol import (
    extract_recent_tool_messages,
    extract_tool_call_ids,
    find_last_real_human_index,
    get_tool_call_name,
    repair_and_validate_tool_results,
)
from react_agent.agent.time import now_iso_in_timezone


logger = logging.getLogger(__name__)


_NON_EXECUTED_RAG_ERROR_CODES = frozenset(
    {
        "GRAPH_STEP_BUDGET_EXHAUSTED",
        "INVALID_TOOL_CALL",
        "RAG_CALL_BUDGET_EXHAUSTED",
        "TOOL_BUDGET_EXHAUSTED",
        "TOOL_DISABLED",
    }
)


# 工具结果裁剪


@dataclass(frozen=True)
class ToolBatch:
    """工具批次的状态与观测投影。"""

    runs: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    error_count: int
    last_ok_result: dict[str, Any] | None
    consecutive_rag_misses: int


def bound_tool_payload(content: str, max_chars: int) -> str:
    """按工具结果信封结构裁剪正文，同时保留状态、错误和溯源元数据。"""
    floor = max(max_chars, 64)
    try:
        payload = decode_tool_result(content)
    except Exception:
        return _trim_text(content, floor)

    meta = payload.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = payload["meta"] = {}

    if payload.get("ok") is False or payload.get("data") is None:
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            if len(error["message"]) > floor:
                error["message"] = _trim_text(error["message"], floor)
                meta["truncated"] = True
        return json.dumps(payload, ensure_ascii=False)

    data = payload.get("data")
    if not isinstance(data, dict):
        return json.dumps(payload, ensure_ascii=False)
    if is_retrieval_payload(payload):
        return bound_retrieval_payload(payload, max_chars)
    results = data.get("results")
    if isinstance(results, list) and results:
        kept: list[Any] = []
        used = 0
        for item in results:
            serialized = json.dumps(item, ensure_ascii=False)
            if used + len(serialized) > max_chars:
                break
            kept.append(item)
            used += len(serialized)
        if len(kept) < len(results):
            data["results"] = kept
            meta.update(
                truncated=True,
                kept_results=len(kept),
                total_results=len(results),
            )
        else:
            meta.setdefault("truncated", False)
    return json.dumps(payload, ensure_ascii=False)


def _trim_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


# 动态工具执行


def bound_tool_messages(messages: list[Any], max_chars: int) -> list[Any]:
    """约束 ToolMessage 正文大小，保留非工具消息原样。"""
    result: list[Any] = []
    for message in messages:
        if isinstance(message, ToolMessage) and isinstance(
            getattr(message, "content", None), str
        ):
            message = message.model_copy(
                update={"content": bound_tool_payload(message.content, max_chars)}
            )
        result.append(message)
    return result


def _validated_tool_messages(
    assistant_message: AIMessage,
    messages: list[Any],
    *,
    max_chars: int,
) -> list[ToolMessage]:
    """在结果进入 State 前统一限幅、补齐并校验工具批次。"""
    bounded = bound_tool_messages(messages, max_chars)
    return repair_and_validate_tool_results(assistant_message, bounded)


async def execute_dynamic_tools(
    state: State,
    config: RunnableConfig,
    dependencies: AgentDependencies,
) -> dict[str, Any]:
    """只执行当前配置允许的工具，并为所有调用生成协议应答。"""
    agent_config = dependencies.config
    active_tools, active_names = dependencies.active_tools()
    retrieval_tool_names = dependencies.tool_names_for(
        ToolCapability.KNOWLEDGE_RETRIEVAL
    )
    try:
        last_message = state.messages[-1] if state.messages else None
        invalid_messages = _invalid_tool_messages(last_message)
        if (
            isinstance(last_message, AIMessage)
            and getattr(last_message, "invalid_tool_calls", None)
            and not last_message.tool_calls
        ):
            logger.warning(
                "检测到 %d 条 invalid_tool_calls，已生成错误 ToolMessage",
                len(invalid_messages),
            )
            return {
                "messages": _validated_tool_messages(
                    last_message,
                    invalid_messages,
                    max_chars=agent_config.max_tool_output_chars,
                )
            }

        disabled_calls, executable_calls = _partition_tool_calls(
            last_message,
            active_names,
        )
        attempted_rag_calls = count_attempted_rag_calls_in_current_turn(
            list(state.messages),
            retrieval_tool_names=retrieval_tool_names,
        )
        allowed_calls, budget_blocked_calls = _apply_rag_call_budget(
            executable_calls,
            attempted=attempted_rag_calls,
            limit=agent_config.rag_call_limit,
            retrieval_tool_names=retrieval_tool_names,
        )
        if not disabled_calls and not budget_blocked_calls:
            node = ToolNode(active_tools, handle_tool_errors=True)
            result = await node.ainvoke(state, config)
            all_messages = invalid_messages + result.get("messages", [])
            result["messages"] = _validated_tool_messages(
                last_message,
                all_messages,
                max_chars=agent_config.max_tool_output_chars,
            )
            return result

        blocked_messages = _blocked_tool_messages(disabled_calls, active_names)
        budget_messages = _rag_budget_messages(
            budget_blocked_calls,
            attempted=attempted_rag_calls,
            limit=agent_config.rag_call_limit,
        )
        executed_messages: list[Any] = []
        if allowed_calls and isinstance(last_message, AIMessage):
            filtered_message = last_message.model_copy(
                update={"tool_calls": allowed_calls}
            )
            transient_messages = list(state.messages[:-1]) + [filtered_message]
            transient_state = replace(state, messages=transient_messages)
            node = ToolNode(active_tools, handle_tool_errors=True)
            invoke_result = await node.ainvoke(transient_state, config)
            executed_messages = invoke_result.get("messages", [])

        update: dict[str, Any] = {
            "messages": _validated_tool_messages(
                last_message,
                invalid_messages
                + blocked_messages
                + budget_messages
                + executed_messages,
                max_chars=agent_config.max_tool_output_chars,
            )
        }
        if budget_blocked_calls:
            update["termination_reason"] = (
                TerminationReason.RAG_CALL_BUDGET_EXHAUSTED.value
            )
        return update
    except Exception as exc:
        return _recover_from_execution_error(state, dependencies, exc)


def _invalid_tool_messages(last_message: Any) -> list[ToolMessage]:
    if not isinstance(last_message, AIMessage):
        return []
    result: list[ToolMessage] = []
    for invalid_call in getattr(last_message, "invalid_tool_calls", None) or []:
        call_id = invalid_call.get("id") or ""
        name = invalid_call.get("name") or "unknown"
        error = invalid_call.get("error") or "参数 JSON 解析失败"
        args = invalid_call.get("args") or ""
        result.append(
            ToolMessage(
                tool_call_id=call_id,
                name=name,
                content=json.dumps(
                    tool_error(
                        tool_name=name,
                        query=str(args)[:200],
                        code="INVALID_TOOL_CALL",
                        message=(
                            f"工具参数 JSON 解析失败：{str(error)[:300]}。"
                            "请检查参数格式后重新调用。"
                        ),
                        meta={"executed": False},
                    ),
                    ensure_ascii=False,
                ),
            )
        )
    return result


def _partition_tool_calls(
    last_message: Any,
    active_names: frozenset[str],
) -> tuple[list[Any], list[Any]]:
    blocked: list[Any] = []
    allowed: list[Any] = []
    if not isinstance(last_message, AIMessage):
        return blocked, allowed
    for tool_call in last_message.tool_calls or []:
        name = get_tool_call_name(tool_call)
        if name and name not in active_names:
            blocked.append(tool_call)
        else:
            allowed.append(tool_call)
    return blocked, allowed


def _apply_rag_call_budget(
    calls: list[Any],
    *,
    attempted: int,
    limit: int,
    retrieval_tool_names: frozenset[str],
) -> tuple[list[Any], list[Any]]:
    """Reserve remaining RAG slots while preserving non-RAG calls."""
    remaining = max(0, limit - attempted)
    allowed: list[Any] = []
    blocked: list[Any] = []
    for tool_call in calls:
        if get_tool_call_name(tool_call) not in retrieval_tool_names:
            allowed.append(tool_call)
        elif remaining > 0:
            allowed.append(tool_call)
            remaining -= 1
        else:
            blocked.append(tool_call)
    return allowed, blocked


def _tool_query(tool_call: Any) -> str:
    args = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
    if isinstance(args, dict):
        return str(args.get("query") or "")[:200]
    return str(args)[:200]


def _rag_budget_messages(
    blocked_calls: list[Any],
    *,
    attempted: int,
    limit: int,
) -> list[ToolMessage]:
    return [
        ToolMessage(
            tool_call_id=tool_call.get("id", ""),
            name=get_tool_call_name(tool_call) or "unknown",
            content=json.dumps(
                tool_error(
                    tool_name=get_tool_call_name(tool_call) or "unknown",
                    query=_tool_query(tool_call),
                    code="RAG_CALL_BUDGET_EXHAUSTED",
                    message="本轮知识库检索次数已达到上限，该调用未执行。",
                    meta={
                        "attempted": attempted,
                        "executed": False,
                        "limit": limit,
                    },
                ),
                ensure_ascii=False,
            ),
        )
        for tool_call in blocked_calls
    ]


def _blocked_tool_messages(
    blocked_calls: list[Any],
    active_names: frozenset[str],
) -> list[ToolMessage]:
    return [
        ToolMessage(
            tool_call_id=tool_call.get("id", ""),
            name=get_tool_call_name(tool_call) or "unknown",
            content=json.dumps(
                tool_error(
                    tool_name=get_tool_call_name(tool_call) or "unknown",
                    query=str(tool_call.get("args", "")),
                    code="TOOL_DISABLED",
                    message=(
                        f"工具 '{get_tool_call_name(tool_call)}' 在当前配置下已被禁用。"
                        f"可用工具：{sorted(active_names)}。"
                    ),
                    meta={"executed": False},
                ),
                ensure_ascii=False,
            ),
        )
        for tool_call in blocked_calls
    ]


def _recover_from_execution_error(
    state: State,
    dependencies: AgentDependencies,
    exc: Exception,
) -> dict[str, Any]:
    logger.error(
        "工具执行层未预期异常：%s: %s",
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    target_index = -1
    required_ids: list[str] = []
    for index in range(len(state.messages) - 1, -1, -1):
        call_ids = extract_tool_call_ids(state.messages[index])
        if call_ids:
            target_index = index
            required_ids = call_ids
            break

    if target_index < 0:
        raise exc

    answered_ids = {
        message.tool_call_id
        for message in state.messages[target_index + 1 :]
        if isinstance(message, ToolMessage)
    }
    missing = [
        ToolMessage(
            tool_call_id=tool_call_id,
            name="unknown",
            content=json.dumps(
                tool_error(
                    tool_name="unknown",
                    query="",
                    code="TOOL_NODE_CRASH",
                    message=f"工具执行层异常：{type(exc).__name__}: {exc}",
                ),
                ensure_ascii=False,
            ),
        )
        for tool_call_id in required_ids
        if tool_call_id not in answered_ids
    ]
    if not missing:
        raise exc

    logger.error("工具执行异常后补齐 %d 条 ToolMessage", len(missing))
    assistant_message = state.messages[target_index]
    if not isinstance(assistant_message, AIMessage):
        raise exc
    return {
        "messages": _validated_tool_messages(
            assistant_message,
            missing,
            max_chars=dependencies.config.max_tool_output_chars,
        )
    }


# 工具批次统计


def _current_turn_rag_messages(
    messages: list[AnyMessage],
    retrieval_tool_names: frozenset[str],
) -> list[ToolMessage]:
    last_human_index = find_last_real_human_index(messages)
    if last_human_index < 0:
        return []
    return [
        message
        for message in messages[last_human_index:]
        if isinstance(message, ToolMessage)
        and getattr(message, "name", None) in retrieval_tool_names
    ]


def _try_decode(message: ToolMessage) -> dict[str, Any] | None:
    try:
        return decode_tool_result(getattr(message, "content", None))
    except (TypeError, ValueError):
        return None


def count_attempted_rag_calls_in_current_turn(
    messages: list[AnyMessage],
    *,
    retrieval_tool_names: frozenset[str],
) -> int:
    """统计当前用户轮次中实际接受执行的 RAG 调用次数。"""
    count = 0
    for message in _current_turn_rag_messages(messages, retrieval_tool_names):
        payload = _try_decode(message)
        if payload is None:
            count += 1
            continue

        meta = payload.get("meta")
        if isinstance(meta, dict) and meta.get("executed") is False:
            continue
        error = payload.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        if code in _NON_EXECUTED_RAG_ERROR_CODES:
            continue
        count += 1
    return count


def count_successful_rag_calls_in_current_turn(
    messages: list[AnyMessage],
    *,
    retrieval_tool_names: frozenset[str],
) -> int:
    """统计当前用户轮次中成功完成的私有知识库检索次数。"""
    count = 0
    for message in _current_turn_rag_messages(messages, retrieval_tool_names):
        payload = _try_decode(message)
        if payload is not None and payload.get("ok") is True:
            count += 1
    return count


def parse_tool_batch(
    messages: list[AnyMessage],
    *,
    timezone: str,
    retrieval_tool_names: frozenset[str],
) -> ToolBatch | None:
    """解析消息尾部工具结果，并计算当前轮连续 RAG 未命中次数。"""
    tool_messages = extract_recent_tool_messages(messages)
    if not tool_messages:
        return None

    runs: list[dict[str, Any]] = []
    last_ok_result: dict[str, Any] | None = None
    for tool_message in tool_messages:
        try:
            payload = decode_tool_result(getattr(tool_message, "content", None))
        except Exception as exc:
            payload = {}
            run_ok: bool | None = False
            run_error: Any = {
                "code": "PARSE_FAILED",
                "message": f"{type(exc).__name__}: {exc}",
            }
        else:
            raw_ok = payload.get("ok")
            run_ok = bool(raw_ok) if raw_ok is not None else None
            run_error = payload.get("error")

        meta = payload.get("meta")
        executed = not (isinstance(meta, dict) and meta.get("executed") is False)

        run = {
            "tool": getattr(tool_message, "name", "") or "unknown_tool",
            "query": payload.get("query"),
            "executed": executed,
            "ok": run_ok,
            "error": run_error,
            "meta": meta,
            "sources": _source_summaries(payload),
            "ts": now_iso_in_timezone(timezone),
        }
        if payload.get("ok") is True:
            last_ok_result = payload
        runs.append(run)

    errors = [run for run in runs if run.get("ok") is False]
    return ToolBatch(
        runs=runs,
        errors=errors,
        error_count=len(errors),
        last_ok_result=last_ok_result,
        consecutive_rag_misses=_count_consecutive_rag_misses(
            messages,
            retrieval_tool_names,
        ),
    )


def _source_summaries(
    payload: dict[str, Any], *, limit: int = 12
) -> list[dict[str, Any]]:
    """为观测面板提取有界来源位置，不复制工具正文。"""
    try:
        outcome = decode_retrieval_outcome(payload)
    except (TypeError, ValueError):
        return []
    if outcome is None:
        return []
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, Any, str]] = set()
    for item in outcome.evidence:
        source = item.source_file
        page = item.source_page
        chunk_id = item.chunk_id
        if not source and page is None and not chunk_id:
            continue
        key = (source, page, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source_file": source or None,
                "source_page": page,
                "chunk_id": chunk_id or None,
            }
        )
        if len(sources) >= limit:
            break
    return sources


def _count_consecutive_rag_misses(
    messages: list[AnyMessage],
    retrieval_tool_names: frozenset[str],
) -> int:
    results: list[bool] = []
    for message in _current_turn_rag_messages(messages, retrieval_tool_names):
        try:
            outcome = decode_retrieval_outcome(getattr(message, "content", None))
        except (TypeError, ValueError):
            continue
        if outcome is None or not outcome.ok:
            continue
        has_content = outcome.has_relevant_content
        if has_content is not None:
            results.append(bool(has_content))

    consecutive_misses = 0
    for has_content in reversed(results):
        if has_content:
            break
        consecutive_misses += 1
    return consecutive_misses


__all__ = [
    "ToolBatch",
    "bound_tool_messages",
    "bound_tool_payload",
    "count_attempted_rag_calls_in_current_turn",
    "count_successful_rag_calls_in_current_turn",
    "execute_dynamic_tools",
    "parse_tool_batch",
]
