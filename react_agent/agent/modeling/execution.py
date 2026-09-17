"""主模型与最终总结模型共享的调用流程。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import State
from react_agent.agent.modeling.history import (
    build_model_input,
    prepare_model_messages,
    sanitize_dangling_tool_calls,
)
from react_agent.metering.model_usage import meter_model_call
from react_agent.agent.support.time import now_iso_in_timezone
from react_agent.agent.tool_flow.calls import get_tool_call_name
from react_agent.agent.tool_flow.catalog import render_tool_catalog


logger = logging.getLogger(__name__)


def _build_system_prompt(
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
    return config.system_prompt.format(
        system_time=system_time,
        language=config.language,
        tool_catalog=tool_catalog,
        consecutive_failure_threshold=config.consecutive_failure_threshold,
    ), active_tools


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
        dependencies,
        tools_enabled=tools_enabled,
    )
    base_model = dependencies.resolve_model()
    model = base_model.bind_tools(active_tools) if tools_enabled else base_model

    messages = prepare_model_messages(list(state.messages), agent_config)
    roles = [getattr(message, "role", type(message).__name__) for message in messages]
    logger.debug("模型输入裁剪后的消息序列: %s", roles)
    messages = sanitize_dangling_tool_calls(messages)

    try:
        response: AIMessage = await model.ainvoke(
            build_model_input(system_prompt, directive, messages),
            config=config,
        )
    except Exception as exc:
        response_body = getattr(getattr(exc, "response", None), "text", None)
        logger.error("模型调用异常类型=%s", type(exc).__name__)
        logger.error("模型调用异常信息=%s", str(exc))
        logger.error("模型响应正文=%s", response_body)
        raise

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
    return response, usage_update


def tool_names_from_response(response: AIMessage) -> list[str]:
    calls = list(response.tool_calls or [])
    calls.extend(getattr(response, "invalid_tool_calls", None) or [])
    return [get_tool_call_name(call) or "unknown" for call in calls]


__all__ = ["invoke_chat_model", "tool_names_from_response"]
