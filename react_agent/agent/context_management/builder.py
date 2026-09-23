"""从完整 Agent State 构造一次模型调用的上下文。"""

from __future__ import annotations

import logging
from typing import Any

from react_agent.agent.context_management.budget_policy import (
    select_context_for_budget,
)
from react_agent.agent.context_management.budgeting import (
    MessageTokenCounter,
    trim_history_to_budget,
)
from react_agent.agent.context_management.compaction import (
    load_valid_summary,
    render_summary,
)
from react_agent.agent.context_management.contracts import (
    ContextBudget,
    ContextBudgetExceeded,
    ContextDiagnostics,
    ModelContext,
)
from react_agent.agent.context_management.evidence import attach_evidence_index
from react_agent.agent.context_management.protocol import validate_tool_exchanges
from react_agent.agent.context_management.segmentation import (
    collapse_consecutive_humans,
    filter_runtime_sentinels,
    latest_turn_ai_messages,
    segment_current_turn,
    segment_turn_blocks,
)
from react_agent.agent.contracts.state import State


logger = logging.getLogger(__name__)


def prepare_model_messages(
    messages: list[Any],
    config: Any,
    *,
    token_counter: MessageTokenCounter | None = None,
    filtered_messages: list[Any] | None = None,
) -> list[Any]:
    """过滤内部消息，并连续保留预算内的完整历史轮次。"""
    filtered = (
        filtered_messages
        if filtered_messages is not None
        else filter_runtime_sentinels(messages)
    )
    if not (config.enable_history_truncation and config.max_history_tokens > 0):
        return collapse_consecutive_humans(filtered)

    trimmed = trim_history_to_budget(
        filtered,
        max_tokens=config.max_history_tokens,
        token_counter=token_counter,
    )
    if not trimmed:
        segments = segment_current_turn(filtered)
        trimmed = list(segments.current or filtered[-1:])
        logger.warning("消息裁剪结果为空，已回退到最近一个完整轮次")
    return collapse_consecutive_humans(trimmed)


def build_model_input(
    system_prompt: str,
    directive: str | None,
    messages: list[Any],
    *,
    summary_text: str | None = None,
    historical_evidence_text: str | None = None,
) -> list[Any]:
    """组合稳定系统提示、瞬态控制指令和已选择的会话消息。"""
    head: list[Any] = [{"role": "system", "content": system_prompt}]
    if directive:
        head.append({"role": "system", "content": directive})
    if summary_text:
        head.append({"role": "assistant", "content": summary_text})
    if historical_evidence_text:
        head.append({"role": "assistant", "content": historical_evidence_text})
    return head + messages


def build_model_context(
    state: State,
    config: Any,
    *,
    system_prompt: str,
    directive: str | None,
    budget: ContextBudget | None = None,
    bound_tools: tuple[Any, ...] = (),
) -> ModelContext:
    """协调消息选择、预算策略与协议校验，返回模型输入及诊断。"""
    source_messages = list(state.messages)
    full_filtered = filter_runtime_sentinels(source_messages)
    summary, summarized_count = load_valid_summary(
        state.conversation_summary, full_filtered
    )
    summary_text = render_summary(summary) if summary else None
    filtered = full_filtered[summarized_count:]
    segments = segment_current_turn(filtered)
    counter = MessageTokenCounter()
    source_tokens = counter.count(filtered)
    selected = prepare_model_messages(
        source_messages,
        config,
        token_counter=counter,
        filtered_messages=filtered,
    )
    history_truncated = bool(
        config.enable_history_truncation
        and config.max_history_tokens > 0
        and source_tokens > config.max_history_tokens
        and len(segment_turn_blocks(segments.completed))
        > len(segment_turn_blocks(segment_current_turn(selected).completed))
    )
    try:
        selection = select_context_for_budget(
            selected,
            counter=counter,
            budget=budget,
            system_prompt=system_prompt,
            directive=directive,
            bound_tools=bound_tools,
            summary_text=summary_text,
            original_completed_count=len(segments.completed),
            turn_evidence=state.turn_evidence,
            turn_evidence_omitted_count=state.turn_evidence_omitted_count,
            conversation_evidence=state.conversation_evidence,
            conversation_evidence_omitted_count=(
                state.conversation_evidence_omitted_count
            ),
        )
    except ContextBudgetExceeded as exc:
        exc.completed_model_messages = tuple(latest_turn_ai_messages(source_messages))
        exc.compaction_usage = state.turn_compaction_usage
        raise

    selected = selection.messages
    summary_text = selection.summary_text
    historical_evidence_text = selection.historical_evidence_text
    budget_report = selection.budget_report
    validate_tool_exchanges(selected)
    model_input = build_model_input(
        system_prompt,
        directive,
        selected,
        summary_text=summary_text,
        historical_evidence_text=historical_evidence_text,
    )
    diagnostics = ContextDiagnostics(
        source_message_count=len(source_messages),
        selected_message_count=len(selected)
        + bool(summary_text)
        + bool(historical_evidence_text),
        estimated_history_tokens=counter.count(selected)
        + (counter.count([{"content": summary_text}]) if summary_text else 0)
        + (
            counter.count([{"content": historical_evidence_text}])
            if historical_evidence_text
            else 0
        ),
        completed_message_count=len(segments.completed),
        current_turn_message_count=len(segments.current),
        evidence_record_count=len(state.turn_evidence),
        evidence_omitted_count=state.turn_evidence_omitted_count,
        historical_evidence_record_count=len(state.conversation_evidence),
        historical_evidence_omitted_count=(
            state.conversation_evidence_omitted_count
            + selection.historical_evidence_omitted_by_budget
        ),
        history_truncated=bool(summarized_count) or history_truncated
        or bool(budget_report and budget_report.dropped_completed_message_count),
        summary_included=bool(summary_text),
        summarized_message_count=summarized_count if summary_text else 0,
    )
    return ModelContext(
        messages=tuple(model_input),
        diagnostics=diagnostics,
        budget_report=budget_report,
    )


__all__ = [
    "attach_evidence_index",
    "build_model_context",
    "build_model_input",
    "prepare_model_messages",
]
