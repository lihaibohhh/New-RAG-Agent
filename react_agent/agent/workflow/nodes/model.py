"""常规模型调用与无工具最终总结节点。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.runtime import Runtime

from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import State
from react_agent.agent.model_execution import (
    invoke_chat_model,
    tool_names_from_response,
)
from react_agent.agent.policies import (
    TerminationReason,
    allow_tool_path,
    model_termination_reason,
)
from react_agent.agent.prompts import render_finalization_directive
from react_agent.agent.tool_flow.budget import (
    count_attempted_rag_calls_in_current_turn,
)
from react_agent.agent.tool_flow.calls import extract_tool_call_ids


logger = logging.getLogger(__name__)


async def call_model(
    state: State,
    runtime: Runtime[AgentDependencies],
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """调用可使用工具的主模型，并根据显式业务预算决定是否收口。"""
    dependencies = runtime.context
    if state.remaining_steps < 1:
        raise GraphRecursionError("call_model 已无可完成的图步骤")
    tools_allowed = allow_tool_path(state.remaining_steps)
    boundary_reason = TerminationReason.GRAPH_STEP_BUDGET_EXHAUSTED.value
    if not tools_allowed:
        logger.info(
            "graph_step_budget_final_answer | remaining_steps=%s",
            state.remaining_steps,
        )
    response, usage_update = await invoke_chat_model(
        state,
        dependencies,
        config,
        allow_tools=tools_allowed,
        directive=(
            state.pending_directive
            if tools_allowed
            else render_finalization_directive(boundary_reason)
        ),
    )
    next_model_round = state.turn_model_rounds + 1
    tool_call_ids = extract_tool_call_ids(response)
    tool_names = tool_names_from_response(response)
    rag_calls = count_attempted_rag_calls_in_current_turn(list(state.messages))
    termination_reason = model_termination_reason(
        dependencies.config,
        tool_names=tool_names if tool_call_ids else [],
        next_model_round=next_model_round,
        completed_tool_batches=state.turn_tool_batches,
        attempted_rag_calls=rag_calls,
    )

    if not tools_allowed and tool_call_ids:
        logger.error(
            "unexpected_tool_call_at_graph_boundary | model_round=%s "
            "tool_batches=%s remaining_steps=%s tool_calls=%s",
            next_model_round,
            state.turn_tool_batches,
            state.remaining_steps,
            tool_names,
        )
        return _emergency_model_boundary_update(
            state,
            response,
            usage_update,
            next_model_round=next_model_round,
            rag_calls=rag_calls,
            tool_names=tool_names,
        )

    update: dict[str, Any] = {
        "messages": [response],
        "step_counter": state.step_counter + 1,
        "turn_model_rounds": next_model_round,
        "termination_reason": (
            termination_reason if tools_allowed else boundary_reason
        ),
        "pending_directive": None,
    }
    update.update(usage_update)
    return update


def _emergency_model_boundary_update(
    state: State,
    response: AIMessage,
    usage_update: dict[str, Any],
    *,
    next_model_round: int,
    rag_calls: int,
    tool_names: list[str],
) -> dict[str, Any]:
    update: dict[str, Any] = {
        "messages": [
            AIMessage(
                id=response.id,
                content=(
                    "本轮处理意外到达系统执行边界，未继续执行新的工具调用。"
                    "现有信息不足以形成可靠结论，请缩小问题范围后重试。"
                ),
                response_metadata=response.response_metadata,
                usage_metadata=response.usage_metadata,
            )
        ],
        "step_counter": state.step_counter + 1,
        "turn_model_rounds": next_model_round,
        "termination_reason": (TerminationReason.UNEXPECTED_RECURSION_BOUNDARY.value),
        "debug_log": [
            f"[call_model] is_last_step=True; "
            f"step_counter={state.step_counter}; "
            f"rag_count={rag_calls}; "
            f"tool_calls={tool_names}; "
            f"last_msg_type={type(state.messages[-1]).__name__}"
        ],
        "pending_directive": None,
    }
    update.update(usage_update)
    return update


async def finalize_model(
    state: State,
    runtime: Runtime[AgentDependencies],
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """调用不绑定工具的最终总结模型。"""
    if state.remaining_steps < 1:
        raise GraphRecursionError("finalize_model 已无可完成的图步骤")
    response, usage_update = await invoke_chat_model(
        state,
        runtime.context,
        config,
        allow_tools=False,
        directive=render_finalization_directive(state.termination_reason),
    )
    if extract_tool_call_ids(response):
        logger.error(
            "finalizer_returned_tool_calls | reason=%s tools=%s",
            state.termination_reason,
            tool_names_from_response(response),
        )
        response = AIMessage(
            id=response.id,
            content=(
                response.content
                or "现有信息不足以形成可靠结论，且本轮工具预算已经结束。"
            ),
            response_metadata=response.response_metadata,
            usage_metadata=response.usage_metadata,
        )

    update: dict[str, Any] = {
        "messages": [response],
        "step_counter": state.step_counter + 1,
        "pending_directive": None,
    }
    update.update(usage_update)
    return update


__all__ = ["call_model", "finalize_model"]
