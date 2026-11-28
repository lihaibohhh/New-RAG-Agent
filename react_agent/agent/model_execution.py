"""主模型与最终总结模型共享的调用流程。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from react_agent.agent.context_management import (
    ContextBudget,
    build_finalization_context,
    build_model_context,
)
from react_agent.agent.context_management.budgeting import (
    observe_token_estimate,
    token_estimate_calibration,
)
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import State
from react_agent.agent.prompts import (
    FINALIZATION_PROTOCOL_FALLBACK,
    render_finalization_retry_directive,
    render_tool_catalog,
)
from react_agent.agent.tool_flow import (
    final_answer_protocol_violation,
    get_tool_call_name,
)
from react_agent.metering import (
    extract_model_usage,
    meter_model_call,
    model_finish_reason,
)
from react_agent.agent.time import now_iso_in_timezone
from react_agent.skills import SkillSelection


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalizerInvocation:
    """一次最终回答生成及其唯一纠正重试结果。"""

    response: AIMessage
    usage_update: dict[str, Any]
    retried: bool = False
    retry_exhausted: bool = False
    violations: tuple[str, ...] = ()


def _build_system_prompt(
    state: State,
    dependencies: AgentDependencies,
    *,
    tools_enabled: bool,
) -> tuple[str, tuple[Any, ...]]:
    config = dependencies.config
    active_tools, _ = dependencies.active_tools()
    tool_catalog = render_tool_catalog(
        active_tools if tools_enabled else (),
        tools_enabled=tools_enabled,
    )
    system_time = now_iso_in_timezone(config.timezone)
    system_prompt = config.system_prompt.format(
        system_time=system_time,
        language=config.language,
        tool_catalog=tool_catalog,
        consecutive_failure_threshold=config.consecutive_failure_threshold,
        rag_soft_call_limit=config.rag_soft_call_limit,
        rag_call_limit=config.rag_call_limit,
        search_soft_call_limit=config.search_soft_call_limit,
        search_call_limit=config.search_call_limit,
        total_tool_call_limit=config.total_tool_call_limit,
    )
    selection = SkillSelection.from_state(state.selected_skill)
    if selection is not None and dependencies.skill_registry is not None:
        skill = dependencies.skill_registry.resolve(selection)
        if skill is None:
            logger.warning(
                "selected_skill_unavailable | name=%s version=%s",
                selection.name,
                selection.version,
            )
        else:
            system_prompt = f"{system_prompt}\n\n{skill.render_for_model()}"
    return system_prompt, active_tools


async def invoke_chat_model(
    state: State,
    dependencies: AgentDependencies,
    config: RunnableConfig | None,
    *,
    allow_tools: bool,
    directive: str | None,
    context_kind: Literal["agent", "finalizer"] = "agent",
    finalizer_attempt: int | None = None,
) -> tuple[AIMessage, dict[str, Any]]:
    """执行一次模型调用，并返回响应和累计用量 State update。"""
    agent_config = dependencies.config
    tools_enabled = bool(agent_config.enable_tools and allow_tools)
    system_prompt, active_tools = _build_system_prompt(
        state,
        dependencies,
        tools_enabled=tools_enabled,
    )
    calibration = token_estimate_calibration(dependencies.model_ref)
    context_budget = ContextBudget(
        max_input_tokens=agent_config.max_input_tokens,
        target_input_tokens=agent_config.model_input_target_tokens,
        reserved_completion_tokens=dependencies.reserved_completion_tokens,
        safety_margin_tokens=agent_config.context_safety_margin_tokens,
        context_window_tokens=dependencies.model_context_window_tokens,
        token_estimate_multiplier=calibration.multiplier,
    )
    if context_kind == "finalizer":
        model_context = build_finalization_context(
            state,
            agent_config,
            system_prompt=system_prompt,
            directive=directive,
            budget=context_budget,
        )
    else:
        model_context = build_model_context(
            state,
            agent_config,
            system_prompt=system_prompt,
            directive=directive,
            budget=context_budget,
            bound_tools=active_tools if tools_enabled else (),
        )
    base_model = dependencies.resolve_model()
    model = base_model.bind_tools(active_tools) if tools_enabled else base_model
    roles = [
        getattr(message, "role", type(message).__name__)
        for message in model_context.messages
    ]
    logger.debug("模型输入裁剪后的消息序列: %s", roles)
    logger.debug(
        "模型上下文诊断: %s",
        model_context.diagnostics,
    )
    logger.debug("模型输入预算: %s", model_context.budget_report)

    call_config = dict(config or {})
    metadata = dict(call_config.get("metadata") or {})
    metadata["model_context_kind"] = context_kind
    if finalizer_attempt is not None:
        metadata["finalizer_attempt"] = finalizer_attempt
    call_config["metadata"] = metadata

    try:
        response: AIMessage = await model.ainvoke(
            model_context.as_list(),
            config=call_config,
        )
    except Exception as exc:
        response_body = getattr(getattr(exc, "response", None), "text", None)
        logger.error("模型调用异常类型=%s", type(exc).__name__)
        logger.error("模型调用异常信息=%s", str(exc))
        logger.error("模型响应正文=%s", response_body)
        raise

    usage = extract_model_usage(response)
    report = model_context.budget_report
    updated_calibration = calibration
    estimate_error_ratio: float | None = None
    if report is not None and usage["prompt_tokens"] > 0:
        updated_calibration = observe_token_estimate(
            dependencies.model_ref,
            estimated_tokens=report.estimated_input_tokens,
            actual_prompt_tokens=usage["prompt_tokens"],
            applied_multiplier=report.token_estimate_multiplier,
        )
        estimate_error_ratio = updated_calibration.last_error_ratio
    level_counts: dict[str, int] = {}
    if report is not None:
        for _, level in report.historical_turn_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
    logger.info(
        "model_call_completed | kind=%s model=%s output_limit_tokens=%s "
        "prompt_tokens=%s output_tokens=%s reasoning_tokens=%s finish_reason=%s "
        "estimated_input_tokens=%s estimate_error_ratio=%s "
        "token_estimate_multiplier=%s compression_level_counts=%s "
        "history_tokens=%s evidence_tokens=%s current_turn_tokens=%s",
        context_kind,
        dependencies.model_ref,
        dependencies.reserved_completion_tokens,
        usage["prompt_tokens"],
        usage["completion_tokens"],
        usage["reasoning_tokens"],
        model_finish_reason(response) or "unknown",
        report.estimated_input_tokens if report else None,
        estimate_error_ratio,
        updated_calibration.multiplier,
        level_counts,
        (
            report.recent_history_tokens + report.summary_tokens
            if report
            else None
        ),
        report.evidence_tokens if report else None,
        (
            report.current_user_tokens + report.current_turn_other_tokens
            if report
            else None
        ),
    )
    usage_update = model_usage_update(state, dependencies, response)
    usage_update["last_context_diagnostics"] = {
        "context_kind": context_kind,
        "estimated_input_tokens": report.estimated_input_tokens if report else None,
        "actual_prompt_tokens": usage["prompt_tokens"],
        "estimate_error_ratio": estimate_error_ratio,
        "token_estimate_multiplier": updated_calibration.multiplier,
        "compression_level_counts": level_counts,
        "history_tokens": (
            report.recent_history_tokens + report.summary_tokens if report else 0
        ),
        "evidence_tokens": report.evidence_tokens if report else 0,
        "current_turn_tokens": (
            report.current_user_tokens + report.current_turn_other_tokens
            if report
            else 0
        ),
        "soft_target_exceeded": report.soft_target_exceeded if report else False,
    }
    return response, usage_update


def _aggregate_response_usage(
    response: AIMessage,
    *responses: AIMessage,
) -> AIMessage:
    """把被丢弃候选的用量附到最终可持久化消息，供非流式计费读取。"""
    usages = [extract_model_usage(message) for message in (*responses, response)]
    combined = {
        key: sum(usage[key] for usage in usages)
        for key in (
            "cache_hit_tokens",
            "cache_miss_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "prompt_tokens",
            "total_tokens",
        )
    }
    response_metadata = dict(response.response_metadata or {})
    response_metadata["token_usage"] = {
        "prompt_tokens": combined["prompt_tokens"],
        "prompt_cache_hit_tokens": combined["cache_hit_tokens"],
        "prompt_cache_miss_tokens": combined["cache_miss_tokens"],
        "completion_tokens": combined["completion_tokens"],
        "total_tokens": combined["total_tokens"],
        "completion_tokens_details": {
            "reasoning_tokens": combined["reasoning_tokens"],
        },
    }
    response_metadata["aggregated_model_calls"] = len(usages)
    return response.model_copy(update={"response_metadata": response_metadata})


async def invoke_finalizer_model(
    state: State,
    dependencies: AgentDependencies,
    config: RunnableConfig | None,
    *,
    directive: str,
) -> FinalizerInvocation:
    """生成最终回答；协议违规时使用瞬态纠正指令重试一次。"""
    first, first_usage = await invoke_chat_model(
        state,
        dependencies,
        config,
        allow_tools=False,
        directive=directive,
        context_kind="finalizer",
        finalizer_attempt=1,
    )
    first_violation = final_answer_protocol_violation(first)
    if first_violation is None:
        return FinalizerInvocation(response=first, usage_update=first_usage)

    logger.warning(
        "finalizer_protocol_violation | attempt=1 reason=%s",
        first_violation,
    )
    retry_state = replace(state, **first_usage)
    correction = render_finalization_retry_directive(first_violation)
    retry, retry_usage = await invoke_chat_model(
        retry_state,
        dependencies,
        config,
        allow_tools=False,
        directive=f"{directive}\n\n{correction}",
        context_kind="finalizer",
        finalizer_attempt=2,
    )
    retry_violation = final_answer_protocol_violation(retry)
    aggregated = _aggregate_response_usage(retry, first)
    if retry_violation is None:
        return FinalizerInvocation(
            response=aggregated,
            usage_update=retry_usage,
            retried=True,
            violations=(first_violation,),
        )

    logger.error(
        "finalizer_protocol_violation | attempt=2 reason=%s fallback=true",
        retry_violation,
    )
    fallback_metadata = dict(aggregated.response_metadata or {})
    fallback_metadata.update(
        finish_reason="protocol_fallback",
        finalizer_protocol_fallback=True,
    )
    fallback = AIMessage(
        id=retry.id,
        content=FINALIZATION_PROTOCOL_FALLBACK,
        response_metadata=fallback_metadata,
        usage_metadata=aggregated.usage_metadata,
    )
    return FinalizerInvocation(
        response=fallback,
        usage_update=retry_usage,
        retried=True,
        retry_exhausted=True,
        violations=(first_violation, retry_violation),
    )


def model_usage_update(
    state: State, dependencies: AgentDependencies, response: AIMessage
) -> dict[str, Any]:
    """主调用与历史压缩调用共用同一用量累计口径。"""
    metered = meter_model_call(
        response,
        model_ref=dependencies.model_ref,
        cost_estimator=dependencies.cost_estimator,
    )
    usage = metered.usage
    model_name = metered.model_name
    cost = metered.cost
    usage_update: dict[str, Any] = {
        "cache_hit_tokens": state.cache_hit_tokens + usage["cache_hit_tokens"],
        "cache_miss_tokens": state.cache_miss_tokens + usage["cache_miss_tokens"],
        "completion_tokens": state.completion_tokens + usage["completion_tokens"],
        "reasoning_tokens": state.reasoning_tokens + usage["reasoning_tokens"],
        "prompt_tokens": state.prompt_tokens + usage["prompt_tokens"],
        "total_tokens": state.total_tokens + usage["total_tokens"],
        "estimated_cost_cny": state.estimated_cost_cny + (cost.amount or 0.0),
        "unpriced_model_count": state.unpriced_model_count
        + (1 if cost.status != "estimated" else 0),
        "pricing_requested_model": dependencies.model_ref,
        "pricing_response_model": str(model_name),
        "pricing_model": cost.billing_model or str(model_name),
        "pricing_tariff": cost.tariff or "",
        "pricing_rate_card_version": cost.rate_card_version,
        "llm_call_count": state.llm_call_count + 1,
    }
    return usage_update


def tool_names_from_response(response: AIMessage) -> list[str]:
    calls = list(response.tool_calls or [])
    calls.extend(getattr(response, "invalid_tool_calls", None) or [])
    return [get_tool_call_name(call) or "unknown" for call in calls]


__all__ = [
    "FinalizerInvocation",
    "invoke_chat_model",
    "invoke_finalizer_model",
    "model_usage_update",
    "tool_names_from_response",
]
