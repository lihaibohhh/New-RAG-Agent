"""工具执行、结果归一化与调用闭合节点。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from react_agent.agent.context_management.evidence import (
    merge_historical_evidence,
    merge_visible_evidence,
)
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import State
from react_agent.agent.policies import (
    TerminationReason,
    tool_batch_termination_reason,
)
from react_agent.agent.tool_flow.calls import close_tool_calls_for_budget
from react_agent.agent.tool_flow.calls import extract_recent_tool_messages
from react_agent.agent.tool_flow.execution import execute_dynamic_tools
from react_agent.agent.tool_flow.observations import parse_tool_batch


async def dynamic_tool_node(
    state: State,
    config: RunnableConfig,
    runtime: Runtime[AgentDependencies],
) -> dict[str, Any]:
    """将 LangGraph 节点调用转交给独立的动态工具执行模块。"""
    return await execute_dynamic_tools(state, config, runtime.context)


async def postprocess_tools(
    state: State,
    runtime: Runtime[AgentDependencies],
) -> dict[str, Any]:
    """将最新工具消息归一化，并应用工具批次终止策略。"""
    agent_config = runtime.context.config
    batch = parse_tool_batch(
        list(state.messages),
        timezone=agent_config.timezone,
    )
    if batch is None:
        return {}

    next_tool_batch = state.turn_tool_batches + 1
    termination_reason = tool_batch_termination_reason(
        agent_config,
        consecutive_rag_misses=batch.consecutive_rag_misses,
        next_tool_batch=next_tool_batch,
        has_errors=bool(batch.errors),
        completed_retries=state.turn_tool_retries,
    )
    if any(
        isinstance(run.get("error"), dict)
        and run["error"].get("code") == "EVIDENCE_PAYLOAD_TOO_LARGE"
        for run in batch.errors
    ):
        termination_reason = TerminationReason.EVIDENCE_OUTPUT_BUDGET_EXHAUSTED.value
    update: dict[str, Any] = {
        "tool_runs": batch.runs,
        "consecutive_failures": batch.consecutive_rag_misses,
        "turn_tool_batches": next_tool_batch,
        "last_tool_batch_errors": batch.errors,
        "termination_reason": termination_reason,
    }
    recent_tool_messages = extract_recent_tool_messages(list(state.messages))
    evidence, omitted = merge_visible_evidence(
        state.turn_evidence,
        recent_tool_messages,
    )
    update["turn_evidence"] = evidence
    update["turn_evidence_omitted_count"] = state.turn_evidence_omitted_count + omitted
    historical, historical_omitted = merge_historical_evidence(
        state.conversation_evidence,
        recent_tool_messages,
    )
    update["conversation_evidence"] = historical
    update["conversation_evidence_omitted_count"] = (
        state.conversation_evidence_omitted_count + historical_omitted
    )
    update["conversation_evidence_scanned_message_count"] = len(state.messages)
    if batch.error_count:
        update["tool_error_count"] = state.tool_error_count + batch.error_count
    if batch.last_ok_result is not None:
        update["last_tool_result"] = {
            key: batch.last_ok_result.get(key)
            for key in ("tool", "query", "ok", "meta")
        }
    return update


async def close_pending_tool_calls(state: State) -> dict[str, Any]:
    """持久化闭合预算耗尽时尚未执行的工具调用。"""
    last_message = state.messages[-1] if state.messages else None
    if not isinstance(last_message, AIMessage):
        return {}

    return {
        "messages": close_tool_calls_for_budget(
            last_message,
            termination_reason=state.termination_reason,
        )
    }


__all__ = [
    "close_pending_tool_calls",
    "dynamic_tool_node",
    "postprocess_tools",
]
