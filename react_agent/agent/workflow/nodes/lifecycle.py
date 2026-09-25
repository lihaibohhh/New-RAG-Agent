"""用户轮次初始化与失败恢复节点。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from react_agent.agent.context_management.compaction import (
    compaction_control_update,
    compaction_messages,
    evaluate_compaction,
    plan_compaction,
    preflight_compaction,
    summary_to_state,
)
from react_agent.agent.context_management.evidence import merge_historical_evidence
from react_agent.agent.context_management.budgeting import estimate_message_tokens
from react_agent.agent.context_management.contracts import ContextBudget
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import State
from react_agent.agent.model_execution import model_usage_update
from react_agent.agent.prompts import render_tool_recovery_directive


logger = logging.getLogger(__name__)


async def prepare_turn(
    state: State,
    runtime: Runtime[AgentDependencies],
) -> dict[str, Any]:
    """在每次新用户输入进入图时重置当前轮业务状态。"""
    selection = None
    if runtime.context.skill_registry is not None:
        selection = runtime.context.skill_registry.select(_latest_human_text(state))
    update: dict[str, Any] = {
        "turn_model_rounds": 0,
        "turn_tool_batches": 0,
        "turn_tool_retries": 0,
        "termination_reason": None,
        "last_tool_batch_errors": [],
        "consecutive_failures": 0,
        "pending_directive": None,
        "last_tool_result": None,
        "turn_evidence": [],
        "turn_evidence_omitted_count": 0,
        "turn_compaction_usage": None,
        "selected_skill": selection.to_state() if selection else None,
    }
    if selection is not None:
        logger.info(
            "skill_selected | name=%s version=%s reason=%s",
            selection.name,
            selection.version,
            selection.reason,
        )
    start = min(
        max(0, state.conversation_evidence_scanned_message_count),
        len(state.messages),
    )
    if start < len(state.messages):
        historical, omitted = merge_historical_evidence(
            state.conversation_evidence,
            [
                message
                for message in state.messages[start:]
                if isinstance(message, ToolMessage)
            ],
        )
        update.update(
            conversation_evidence=historical,
            conversation_evidence_omitted_count=(
                state.conversation_evidence_omitted_count + omitted
            ),
            conversation_evidence_scanned_message_count=len(state.messages),
        )
    return update


def _latest_human_text(state: State) -> str:
    for message in reversed(state.messages):
        if not isinstance(message, HumanMessage):
            continue
        if isinstance(message.content, str):
            return message.content
        text_parts: list[str] = []
        for block in message.content or []:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        return "\n".join(text_parts)
    return ""


async def compact_history(
    state: State,
    runtime: Runtime[AgentDependencies],
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """每个新用户轮最多压缩一批较早完整轮次，失败时沿用原历史。"""
    dependencies = runtime.context
    plan = plan_compaction(
        state,
        dependencies.config,
        context_window_tokens=dependencies.model_context_window_tokens,
    )
    if plan is None:
        return {}
    retry_new_tokens = dependencies.config.history_compaction_retry_new_tokens
    preflight_reason = preflight_compaction(plan)
    if preflight_reason:
        logger.info(
            "history_compaction_deferred | reason=%s source_tokens=%s "
            "projected_floor_tokens=%s",
            preflight_reason,
            plan.source_tokens,
            plan.projected_floor_tokens,
        )
        return {
            "compaction_control": compaction_control_update(
                plan,
                status="deferred",
                reason=preflight_reason,
                retry_new_tokens=retry_new_tokens,
            )
        }
    model_input = compaction_messages(plan)
    budget = ContextBudget(
        max_input_tokens=dependencies.config.max_input_tokens,
        reserved_completion_tokens=dependencies.reserved_completion_tokens,
        safety_margin_tokens=dependencies.config.context_safety_margin_tokens,
        context_window_tokens=dependencies.model_context_window_tokens,
    )
    if estimate_message_tokens(model_input) > budget.input_limit_tokens:
        reason = "input_budget_exceeded"
        logger.warning("history_compaction_rejected | reason=%s", reason)
        return {
            "compaction_control": compaction_control_update(
                plan,
                status="rejected",
                reason=reason,
                retry_new_tokens=retry_new_tokens,
            )
        }
    model_config = dict(config or {})
    model_config["tags"] = list(model_config.get("tags") or []) + [
        "context_compaction"
    ]
    try:
        compaction_model = dependencies.resolve_model()
        bind = getattr(compaction_model, "bind", None)
        if callable(bind):
            compaction_model = bind(
                max_tokens=dependencies.config.history_compaction_max_output_tokens
            )
        response = await compaction_model.ainvoke(
            model_input, config=model_config
        )
    except Exception as exc:
        reason = f"model_error:{type(exc).__name__}"
        logger.warning("history_compaction_rejected | reason=%s", reason)
        return {
            "compaction_control": compaction_control_update(
                plan,
                status="rejected",
                reason=reason,
                retry_new_tokens=retry_new_tokens,
            )
        }
    if not isinstance(response, AIMessage):
        reason = "invalid_response_type"
        logger.warning("history_compaction_rejected | reason=%s", reason)
        return {
            "compaction_control": compaction_control_update(
                plan,
                status="rejected",
                reason=reason,
                retry_new_tokens=retry_new_tokens,
            )
        }
    update = model_usage_update(state, dependencies, response)
    update["turn_compaction_usage"] = {
        "model_name": update["pricing_response_model"],
        "prompt_tokens": update["prompt_tokens"] - state.prompt_tokens,
        "completion_tokens": update["completion_tokens"] - state.completion_tokens,
        "total_tokens": update["total_tokens"] - state.total_tokens,
        "cost_cny": update["estimated_cost_cny"] - state.estimated_cost_cny,
        "priced": update["unpriced_model_count"] == state.unpriced_model_count,
    }
    evaluation = (
        evaluate_compaction(plan, response.content)
        if isinstance(response.content, str)
        else None
    )
    if evaluation is None or not evaluation.accepted:
        reason = evaluation.reason if evaluation else "invalid_response_content"
        update["turn_compaction_usage"]["outcome"] = reason
        update["compaction_control"] = compaction_control_update(
            plan,
            status="rejected",
            reason=reason,
            retry_new_tokens=retry_new_tokens,
        )
        logger.warning(
            "history_compaction_rejected | reason=%s source_tokens=%s "
            "summary_tokens=%s retry_after_source_tokens=%s",
            reason,
            plan.source_tokens,
            evaluation.summary_tokens if evaluation else None,
            update["compaction_control"]["retry_after_source_tokens"],
        )
        return update
    summary = evaluation.summary
    assert summary is not None
    update["turn_compaction_usage"]["outcome"] = "accepted"
    update["conversation_summary"] = summary_to_state(summary)
    update["compaction_control"] = compaction_control_update(
        plan,
        status="accepted",
        reason="accepted",
        retry_new_tokens=retry_new_tokens,
    )
    logger.info(
        "history_compaction_accepted | source_tokens=%s summary_tokens=%s "
        "source_message_count=%s",
        evaluation.source_tokens,
        evaluation.summary_tokens,
        len(plan.source_message_ids),
    )
    return update


async def reflection_node(state: State) -> dict[str, Any]:
    """将最近工具批次错误转换为一次受预算约束的恢复指令。"""
    if not state.last_tool_batch_errors:
        return {}
    logger.warning(
        "tool_batch_recovery | errors=%s retry=%s",
        len(state.last_tool_batch_errors),
        state.turn_tool_retries + 1,
    )
    return {
        "pending_directive": render_tool_recovery_directive(
            state.last_tool_batch_errors
        ),
        "turn_tool_retries": state.turn_tool_retries + 1,
    }


__all__ = ["compact_history", "prepare_turn", "reflection_node"]
