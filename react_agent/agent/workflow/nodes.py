"""Agent 工作流节点。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.runtime import Runtime

from react_agent.agent.context_management.compaction import (
    compaction_control_update,
    compaction_messages,
    evaluate_compaction,
    plan_compaction,
    preflight_compaction,
    summary_to_state,
)
from react_agent.metering import (
    extract_model_usage,
    is_output_truncated,
    model_finish_reason,
)
from react_agent.agent.context_management.evidence import (
    merge_historical_evidence,
    merge_visible_evidence,
)
from react_agent.agent.context_management.budgeting import estimate_message_tokens
from react_agent.agent.context_management.contracts import ContextBudget
from react_agent.agent.contracts.dependencies import (
    AgentDependencies,
    ToolCapability,
)
from react_agent.agent.contracts.state import State
from react_agent.agent.model_execution import (
    invoke_chat_model,
    invoke_finalizer_model,
    model_usage_update,
    tool_names_from_response,
)
from react_agent.agent.policies import (
    TerminationReason,
    allow_tool_path,
    can_execute_tools,
    can_schedule_finalizer,
    finalize_after_tools,
    model_termination_reason,
    tool_batch_termination_reason,
)
from react_agent.agent.prompts import (
    render_finalization_directive,
    render_tool_recovery_directive,
)
from react_agent.agent.tool_flow import (
    close_tool_calls_for_budget,
    count_attempted_rag_calls_in_current_turn,
    execute_dynamic_tools,
    extract_recent_tool_messages,
    extract_tool_call_ids,
    parse_tool_batch,
    repair_and_validate_tool_results,
)


logger = logging.getLogger(__name__)


# 用户轮次生命周期


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
        "turn_tool_runs": [],
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
            retrieval_tool_names=runtime.context.tool_names_for(
                ToolCapability.KNOWLEDGE_RETRIEVAL
            ),
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
        reserved_completion_tokens=(
            dependencies.config.history_compaction_max_output_tokens
        ),
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
        compaction_model = dependencies.limit_model_output(
            dependencies.resolve_model(),
            max_tokens=dependencies.config.history_compaction_max_output_tokens,
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
    response_usage = extract_model_usage(response)
    logger.info(
        "model_call_completed | kind=context_compaction model=%s "
        "output_limit_tokens=%s prompt_tokens=%s output_tokens=%s "
        "reasoning_tokens=%s finish_reason=%s",
        dependencies.model_ref,
        dependencies.config.history_compaction_max_output_tokens,
        response_usage["prompt_tokens"],
        response_usage["completion_tokens"],
        response_usage["reasoning_tokens"],
        model_finish_reason(response) or "unknown",
    )
    update = model_usage_update(state, dependencies, response)
    update["turn_compaction_usage"] = {
        "model_name": update["pricing_response_model"],
        "prompt_tokens": update["prompt_tokens"] - state.prompt_tokens,
        "completion_tokens": update["completion_tokens"] - state.completion_tokens,
        "total_tokens": update["total_tokens"] - state.total_tokens,
        "cost_cny": update["estimated_cost_cny"] - state.estimated_cost_cny,
        "priced": update["unpriced_model_count"] == state.unpriced_model_count,
    }
    if is_output_truncated(response):
        reason = "output_truncated"
        update["turn_compaction_usage"]["outcome"] = reason
        update["compaction_control"] = compaction_control_update(
            plan,
            status="rejected",
            reason=reason,
            retry_new_tokens=retry_new_tokens,
        )
        logger.warning(
            "history_compaction_rejected | reason=%s source_tokens=%s",
            reason,
            plan.source_tokens,
        )
        return update
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


# 模型调用


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
    finalizer_result = None
    if tools_allowed:
        response, usage_update = await invoke_chat_model(
            state,
            dependencies,
            config,
            allow_tools=True,
            directive=state.pending_directive,
            context_kind="agent",
        )
    else:
        finalizer_result = await invoke_finalizer_model(
            state,
            dependencies,
            config,
            directive=render_finalization_directive(boundary_reason),
        )
        response = finalizer_result.response
        usage_update = finalizer_result.usage_update
    next_model_round = state.turn_model_rounds + 1
    if is_output_truncated(response):
        logger.warning(
            "model_output_truncated | node=call_model model_round=%s",
            next_model_round,
        )
        update: dict[str, Any] = {
            "messages": [_truncated_response(response)],
            "step_counter": state.step_counter + 1,
            "turn_model_rounds": next_model_round,
            "termination_reason": TerminationReason.MODEL_OUTPUT_TRUNCATED.value,
            "pending_directive": None,
        }
        update.update(usage_update)
        return update
    tool_call_ids = extract_tool_call_ids(response)
    tool_names = tool_names_from_response(response)
    retrieval_tool_names = dependencies.tool_names_for(
        ToolCapability.KNOWLEDGE_RETRIEVAL
    )
    rag_calls = count_attempted_rag_calls_in_current_turn(
        list(state.messages),
        retrieval_tool_names=retrieval_tool_names,
    )
    termination_reason = model_termination_reason(
        dependencies.config,
        tool_names=tool_names if tool_call_ids else [],
        retrieval_tool_names=retrieval_tool_names,
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

    update = {
        "messages": [response],
        "step_counter": state.step_counter + 1,
        "turn_model_rounds": next_model_round,
        "termination_reason": (
            termination_reason
            if tools_allowed
            else (
                TerminationReason.FINALIZER_PROTOCOL_RETRY_EXHAUSTED.value
                if finalizer_result and finalizer_result.retry_exhausted
                else boundary_reason
            )
        ),
        "pending_directive": None,
    }
    if finalizer_result and finalizer_result.retried:
        update["debug_log"] = [
            "[call_model] finalizer_protocol_retry; "
            f"violations={list(finalizer_result.violations)}; "
            f"fallback={finalizer_result.retry_exhausted}"
        ]
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
        "termination_reason": TerminationReason.UNEXPECTED_RECURSION_BOUNDARY.value,
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
    finalizer_result = await invoke_finalizer_model(
        state,
        runtime.context,
        config,
        directive=render_finalization_directive(state.termination_reason),
    )
    response = finalizer_result.response
    usage_update = finalizer_result.usage_update
    if is_output_truncated(response):
        logger.warning("model_output_truncated | node=finalize_model")
        response = _truncated_response(response)

    update: dict[str, Any] = {
        "messages": [response],
        "step_counter": state.step_counter + 1,
        "pending_directive": None,
    }
    if finalizer_result.retry_exhausted:
        update["termination_reason"] = (
            TerminationReason.FINALIZER_PROTOCOL_RETRY_EXHAUSTED.value
        )
    if finalizer_result.retried:
        update["debug_log"] = [
            "[finalize_model] finalizer_protocol_retry; "
            f"violations={list(finalizer_result.violations)}; "
            f"fallback={finalizer_result.retry_exhausted}"
        ]
    update.update(usage_update)
    return update


def _truncated_response(response: AIMessage) -> AIMessage:
    """丢弃可能不完整的工具调用，并给出明确结果。"""
    partial = response.content.strip() if isinstance(response.content, str) else ""
    notice = (
        "模型输出达到长度上限，本次结果可能不完整。"
        "请缩小问题范围，或提高模型输出 Token 上限后重试。"
    )
    content = f"{partial}\n\n{notice}" if partial else notice
    return AIMessage(
        id=response.id,
        content=content,
        response_metadata=response.response_metadata,
        usage_metadata=response.usage_metadata,
    )


# 工具节点


async def dynamic_tool_node(
    state: State,
    config: RunnableConfig,
    runtime: Runtime[AgentDependencies],
) -> dict[str, Any]:
    """检查图步骤预算后，将调用交给动态工具执行模块。"""
    if not can_execute_tools(state.remaining_steps):
        if state.remaining_steps < 2:
            raise GraphRecursionError("tools 已无足够图步骤闭合工具调用")
        last_message = state.messages[-1] if state.messages else None
        if not isinstance(last_message, AIMessage):
            raise GraphRecursionError("tools 无法识别待闭合的模型工具调用")
        logger.error(
            "unexpected_recursion_boundary | node=tools remaining_steps=%s",
            state.remaining_steps,
        )
        closed = close_tool_calls_for_budget(
            last_message,
            termination_reason=TerminationReason.UNEXPECTED_RECURSION_BOUNDARY.value,
            error_code="GRAPH_STEP_BUDGET_EXHAUSTED",
            error_message="本轮执行步数不足，该工具未执行。",
        )
        return {
            "messages": repair_and_validate_tool_results(last_message, closed),
            "termination_reason": TerminationReason.UNEXPECTED_RECURSION_BOUNDARY.value,
        }
    return await execute_dynamic_tools(state, config, runtime.context)


async def postprocess_tools(
    state: State,
    runtime: Runtime[AgentDependencies],
) -> dict[str, Any]:
    """将最新工具消息归一化，并应用工具批次终止策略。"""
    agent_config = runtime.context.config
    retrieval_tool_names = runtime.context.tool_names_for(
        ToolCapability.KNOWLEDGE_RETRIEVAL
    )
    batch = parse_tool_batch(
        list(state.messages),
        timezone=agent_config.timezone,
        retrieval_tool_names=retrieval_tool_names,
    )
    if batch is None:
        return {}

    next_tool_batch = state.turn_tool_batches + 1
    termination_reason = state.termination_reason
    if termination_reason is None:
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
    ) and termination_reason is None:
        termination_reason = TerminationReason.EVIDENCE_OUTPUT_BUDGET_EXHAUSTED.value
    if finalize_after_tools(state.remaining_steps) and termination_reason is None:
        termination_reason = TerminationReason.GRAPH_STEP_BUDGET_EXHAUSTED.value
    update: dict[str, Any] = {
        "tool_runs": batch.runs,
        "turn_tool_runs": [*state.turn_tool_runs, *batch.runs],
        "tool_run_count": state.tool_run_count + len(batch.runs),
        "consecutive_failures": batch.consecutive_rag_misses,
        "turn_tool_batches": next_tool_batch,
        "last_tool_batch_errors": batch.errors,
        "termination_reason": termination_reason,
    }
    if not can_schedule_finalizer(state.remaining_steps):
        logger.error(
            "unexpected_recursion_boundary | node=postprocess_tools remaining_steps=%s",
            state.remaining_steps,
        )
        update["termination_reason"] = (
            TerminationReason.UNEXPECTED_RECURSION_BOUNDARY.value
        )
        update["messages"] = [
            AIMessage(
                content=(
                    "本轮处理意外到达系统执行边界，无法继续生成可靠结论。"
                    "请缩小问题范围后重试。"
                )
            )
        ]
    recent_tool_messages = extract_recent_tool_messages(list(state.messages))
    evidence, omitted = merge_visible_evidence(
        state.turn_evidence,
        recent_tool_messages,
        retrieval_tool_names=retrieval_tool_names,
    )
    update["turn_evidence"] = evidence
    update["turn_evidence_omitted_count"] = state.turn_evidence_omitted_count + omitted
    historical, historical_omitted = merge_historical_evidence(
        state.conversation_evidence,
        recent_tool_messages,
        retrieval_tool_names=retrieval_tool_names,
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

    kwargs: dict[str, str] = {}
    if state.termination_reason == TerminationReason.RAG_CALL_BUDGET_EXHAUSTED.value:
        kwargs = {
            "error_code": "RAG_CALL_BUDGET_EXHAUSTED",
            "error_message": "本轮知识库检索次数已达到上限，该调用未执行。",
        }
    closed = close_tool_calls_for_budget(
        last_message,
        termination_reason=state.termination_reason,
        **kwargs,
    )
    return {
        "messages": repair_and_validate_tool_results(last_message, closed)
    }


__all__ = [
    "call_model",
    "compact_history",
    "close_pending_tool_calls",
    "dynamic_tool_node",
    "finalize_model",
    "postprocess_tools",
    "prepare_turn",
    "reflection_node",
]
