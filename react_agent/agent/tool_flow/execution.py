"""动态工具权限、执行与工具调用协议补齐。"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import State
from react_agent.agent.policies import TerminationReason
from react_agent.agent.tool_flow.payload import bound_tool_payload
from react_agent.agent.tool_flow.protocol import (
    extract_tool_call_ids,
    get_tool_call_name,
)
from react_agent.agent.tool_flow.results import (
    count_attempted_rag_calls_in_current_turn,
)
from react_agent.tooling.results import tool_error


logger = logging.getLogger(__name__)
_RAG_TOOL_NAME = "query_internal_knowledge"


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


async def execute_dynamic_tools(
    state: State,
    config: RunnableConfig,
    dependencies: AgentDependencies,
) -> dict[str, Any]:
    """只执行当前配置允许的工具，并为所有调用生成协议应答。"""
    agent_config = dependencies.config
    active_tools, active_names = dependencies.active_tools()
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
                "messages": bound_tool_messages(
                    invalid_messages,
                    agent_config.max_tool_output_chars,
                )
            }

        disabled_calls, executable_calls = _partition_tool_calls(
            last_message,
            active_names,
        )
        attempted_rag_calls = count_attempted_rag_calls_in_current_turn(
            list(state.messages)
        )
        allowed_calls, budget_blocked_calls = _apply_rag_call_budget(
            executable_calls,
            attempted=attempted_rag_calls,
            limit=agent_config.rag_call_limit,
        )
        if not disabled_calls and not budget_blocked_calls:
            node = ToolNode(active_tools, handle_tool_errors=True)
            result = await node.ainvoke(state, config)
            all_messages = invalid_messages + result.get("messages", [])
            result["messages"] = bound_tool_messages(
                all_messages,
                agent_config.max_tool_output_chars,
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
            "messages": bound_tool_messages(
                invalid_messages
                + blocked_messages
                + budget_messages
                + executed_messages,
                agent_config.max_tool_output_chars,
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
) -> tuple[list[Any], list[Any]]:
    """Reserve remaining RAG slots while preserving non-RAG calls."""
    remaining = max(0, limit - attempted)
    allowed: list[Any] = []
    blocked: list[Any] = []
    for tool_call in calls:
        if get_tool_call_name(tool_call) != _RAG_TOOL_NAME:
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
            name=_RAG_TOOL_NAME,
            content=json.dumps(
                tool_error(
                    tool_name=_RAG_TOOL_NAME,
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
    return {
        "messages": bound_tool_messages(
            missing,
            dependencies.config.max_tool_output_chars,
        )
    }


__all__ = ["bound_tool_messages", "execute_dynamic_tools"]
