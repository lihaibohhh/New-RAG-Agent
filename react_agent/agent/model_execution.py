"""主模型与最终总结模型共享的调用流程。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from react_agent.agent.context_management import ContextBudget, build_model_context
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import State
from react_agent.agent.prompts import render_tool_catalog
from react_agent.agent.tool_flow import get_tool_call_name
from react_agent.metering.model_usage import (
    extract_model_usage,
    meter_model_call,
    model_finish_reason,
)
from react_agent.agent.time import now_iso_in_timezone
from react_agent.skills import SkillSelection


logger = logging.getLogger(__name__)


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
) -> tuple[AIMessage, dict[str, Any]]:
    """执行一次模型调用，并返回响应和累计用量 State update。"""
    agent_config = dependencies.config
    tools_enabled = bool(agent_config.enable_tools and allow_tools)
    system_prompt, active_tools = _build_system_prompt(
        state,
        dependencies,
        tools_enabled=tools_enabled,
    )
    model_context = build_model_context(
        state,
        agent_config,
        system_prompt=system_prompt,
        directive=directive,
        budget=ContextBudget(
            max_input_tokens=agent_config.max_input_tokens,
            reserved_completion_tokens=dependencies.reserved_completion_tokens,
            safety_margin_tokens=agent_config.context_safety_margin_tokens,
            context_window_tokens=dependencies.model_context_window_tokens,
        ),
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

    try:
        response: AIMessage = await model.ainvoke(
            model_context.as_list(),
            config=config,
        )
    except Exception as exc:
        response_body = getattr(getattr(exc, "response", None), "text", None)
        logger.error("模型调用异常类型=%s", type(exc).__name__)
        logger.error("模型调用异常信息=%s", str(exc))
        logger.error("模型响应正文=%s", response_body)
        raise

    usage = extract_model_usage(response)
    logger.info(
        "model_call_completed | kind=%s model=%s output_limit_tokens=%s "
        "prompt_tokens=%s output_tokens=%s reasoning_tokens=%s finish_reason=%s",
        "agent" if tools_enabled else "finalizer",
        dependencies.model_ref,
        dependencies.reserved_completion_tokens,
        usage["prompt_tokens"],
        usage["completion_tokens"],
        usage["reasoning_tokens"],
        model_finish_reason(response) or "unknown",
    )

    return response, model_usage_update(state, dependencies, response)


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


__all__ = ["invoke_chat_model", "model_usage_update", "tool_names_from_response"]
