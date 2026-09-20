"""从完整 Agent State 构造一次模型调用的上下文。"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from langchain_core.messages import ToolMessage

from react_agent.agent.context_management.budgeting import (
    MessageTokenCounter,
    estimate_tool_schema_tokens,
    trim_history_to_budget,
)
from react_agent.agent.context_management.compaction import (
    load_valid_summary,
    render_summary,
)
from react_agent.agent.context_management.contracts import (
    BudgetReport,
    ContextBudget,
    ContextBudgetExceeded,
    ContextDiagnostics,
    ModelContext,
)
from react_agent.agent.context_management.evidence import (
    render_evidence_index,
    render_historical_evidence_index,
)
from react_agent.agent.context_management.projection import (
    project_completed_tool_messages,
    project_current_tool_messages,
)
from react_agent.agent.context_management.protocol import (
    sanitize_dangling_tool_calls,
    validate_tool_exchanges,
)
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


def attach_evidence_index(messages: list[Any], evidence_index: str) -> list[Any]:
    """仅在本次模型输入的工具角色中附加证据索引，不改写 Checkpoint。"""
    if not evidence_index:
        return messages
    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        message = result[index]
        if isinstance(message, ToolMessage) and isinstance(message.content, str):
            result[index] = message.model_copy(
                update={"content": f"{message.content}\n\n{evidence_index}"}
            )
            return result
    logger.warning("存在本轮证据索引，但模型输入中没有可承载索引的工具消息")
    return result


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


def _drop_oldest_completed_turn(messages: list[Any]) -> list[Any] | None:
    """只移除已完成历史的最早一轮，绝不删除当前用户轮次。"""
    segments = segment_current_turn(messages)
    completed = list(segments.completed)
    if not completed:
        return None
    blocks = segment_turn_blocks(completed)
    return [
        message for block in blocks[1:] for message in block.messages
    ] + list(segments.current)


def _budget_report(
    selected: list[Any],
    projected: list[Any],
    *,
    counter: MessageTokenCounter,
    budget: ContextBudget,
    system_tokens: int,
    directive_tokens: int,
    tool_schema_tokens: int,
    summary_tokens: int,
    evidence_tokens: int,
    historical_evidence_tokens: int,
    original_completed_count: int,
    projected_tool_message_count: int = 0,
    tool_content_chars_removed: int = 0,
    evidence_excerpt_chars_removed: int = 0,
    evidence_records_omitted_by_budget: int = 0,
) -> BudgetReport:
    segments = segment_current_turn(projected)
    current = list(segments.current)
    recent_history_tokens = counter.count(list(segments.completed))
    current_user_tokens = counter.count(current[:1])
    current_turn_other_tokens = counter.count(current[1:])
    estimated_input_tokens = (
        system_tokens
        + directive_tokens
        + tool_schema_tokens
        + summary_tokens
        + historical_evidence_tokens
        + recent_history_tokens
        + current_user_tokens
        + current_turn_other_tokens
    )
    projection_reasons = tuple(
        reason
        for applied, reason in (
            (tool_content_chars_removed > 0, "tool_payload_budget"),
            (evidence_excerpt_chars_removed > 0, "evidence_excerpt_budget"),
            (evidence_records_omitted_by_budget > 0, "evidence_record_budget"),
        )
        if applied
    )
    return BudgetReport(
        input_limit_tokens=budget.input_limit_tokens,
        estimated_input_tokens=estimated_input_tokens,
        system_tokens=system_tokens,
        directive_tokens=directive_tokens,
        tool_schema_tokens=tool_schema_tokens,
        current_user_tokens=current_user_tokens,
        current_turn_other_tokens=current_turn_other_tokens,
        evidence_tokens=evidence_tokens + historical_evidence_tokens,
        recent_history_tokens=recent_history_tokens,
        summary_tokens=summary_tokens,
        historical_evidence_tokens=historical_evidence_tokens,
        reserved_completion_tokens=budget.reserved_completion_tokens,
        safety_margin_tokens=budget.safety_margin_tokens,
        context_window_tokens=budget.context_window_tokens,
        dropped_completed_message_count=max(
            0,
            original_completed_count - len(segment_current_turn(selected).completed),
        ),
        projected_tool_message_count=projected_tool_message_count,
        tool_content_chars_removed=tool_content_chars_removed,
        evidence_excerpt_chars_removed=evidence_excerpt_chars_removed,
        evidence_records_omitted_by_budget=evidence_records_omitted_by_budget,
        projection_reasons=projection_reasons,
    )


def _projected_candidates(
    messages: list[Any],
    records: list[dict[str, Any]],
    omitted_count: int,
):
    """先缩减重复证据摘录，再缩减工具正文，最后缩减索引条数。"""
    original_excerpt_chars = sum(len(str(record.get("excerpt", ""))) for record in records)
    specs = [
        (None, excerpt_chars, None)
        for excerpt_chars in (120, 60, 0)
    ]
    specs.extend((tool_chars, 0, None) for tool_chars in (2048, 1024, 512, 256, 0))
    specs.extend((0, 0, count) for count in (12, 8, 4, 2, 1, 0))
    for tool_chars, excerpt_chars, max_records in specs:
        if tool_chars is None:
            base = messages
            projected_count = removed_chars = 0
        else:
            base, projected_count, removed_chars = project_current_tool_messages(
                messages, max_chars=tool_chars
            )
        visible_records = records if max_records is None else records[:max_records]
        visible_excerpt_chars = sum(
            len(str(record.get("excerpt", ""))[:excerpt_chars])
            for record in visible_records
        )
        index = render_evidence_index(
            records,
            omitted_count=omitted_count,
            max_records=max_records,
            max_excerpt_chars=excerpt_chars,
        )
        yield (
            base,
            attach_evidence_index(base, index),
            projected_count,
            removed_chars,
            original_excerpt_chars - visible_excerpt_chars,
            len(records) - len(visible_records),
        )


def build_model_context(
    state: State,
    config: Any,
    *,
    system_prompt: str,
    directive: str | None,
    budget: ContextBudget | None = None,
    bound_tools: tuple[Any, ...] = (),
) -> ModelContext:
    """执行上下文选择、协议修复和证据注入，并返回诊断信息。"""
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
    evidence_records = state.turn_evidence
    evidence_index = render_evidence_index(
        evidence_records,
        omitted_count=state.turn_evidence_omitted_count,
    )
    historical_evidence_limit = len(state.conversation_evidence)
    historical_evidence_text = ""
    historical_evidence_tokens = 0
    historical_evidence_omitted_by_budget = 0
    budget_report: BudgetReport | None = None
    if budget is not None:
        system_tokens = counter.count([{"role": "system", "content": system_prompt}])
        directive_tokens = (
            counter.count([{"role": "system", "content": directive}])
            if directive
            else 0
        )
        tool_schema_tokens = estimate_tool_schema_tokens(
            bound_tools, token_counter=counter
        )

    while True:
        repaired = sanitize_dangling_tool_calls(selected)
        selected_tool_ids = {
            str(getattr(message, "id", None) or "")
            for message in repaired
            if isinstance(message, ToolMessage) and getattr(message, "id", None)
        }
        selected_tool_call_ids = {
            str(getattr(message, "tool_call_id", None) or "")
            for message in repaired
            if isinstance(message, ToolMessage)
            and getattr(message, "tool_call_id", None)
        }
        historical_records = [
            record
            for record in state.conversation_evidence
            if str(record.get("last_tool_message_id") or "") not in selected_tool_ids
            and str(record.get("last_tool_call_id") or "")
            not in selected_tool_call_ids
        ]
        if historical_evidence_limit > 0 and historical_records:
            historical_evidence_text = render_historical_evidence_index(
                historical_records,
                omitted_count=state.conversation_evidence_omitted_count,
                max_records=historical_evidence_limit,
            )
        else:
            historical_evidence_text = ""
        historical_evidence_omitted_by_budget = max(
            0, len(historical_records) - historical_evidence_limit
        )
        projected = attach_evidence_index(repaired, evidence_index)
        if budget is None:
            selected = projected
            break

        evidence_tokens = counter.count(projected) - counter.count(repaired)
        historical_evidence_tokens = (
            counter.count(
                [{"role": "assistant", "content": historical_evidence_text}]
            )
            if historical_evidence_text
            else 0
        )
        summary_tokens = (
            counter.count([{"role": "assistant", "content": summary_text}])
            if summary_text
            else 0
        )
        budget_report = _budget_report(
            selected,
            projected,
            counter=counter,
            budget=budget,
            system_tokens=system_tokens,
            directive_tokens=directive_tokens,
            tool_schema_tokens=tool_schema_tokens,
            summary_tokens=summary_tokens,
            evidence_tokens=evidence_tokens,
            historical_evidence_tokens=historical_evidence_tokens,
            original_completed_count=len(segments.completed),
        )
        fixed_tokens = system_tokens + directive_tokens + tool_schema_tokens
        if fixed_tokens > budget.input_limit_tokens:
            reason = "FIXED_CONTEXT_TOO_LARGE"
        elif fixed_tokens + budget_report.current_user_tokens > budget.input_limit_tokens:
            reason = "USER_INPUT_TOO_LARGE"
        elif budget_report.estimated_input_tokens <= budget.input_limit_tokens:
            selected = projected
            break
        else:
            if segment_current_turn(selected).completed:
                for tool_chars in (2048, 1024, 512, 256, 0):
                    old_base, projected_count, removed_chars = (
                        project_completed_tool_messages(
                            repaired, max_chars=tool_chars
                        )
                    )
                    if not projected_count:
                        continue
                    old_candidate = attach_evidence_index(old_base, evidence_index)
                    candidate_report = _budget_report(
                        selected,
                        old_candidate,
                        counter=counter,
                        budget=budget,
                        system_tokens=system_tokens,
                        directive_tokens=directive_tokens,
                        tool_schema_tokens=tool_schema_tokens,
                        summary_tokens=summary_tokens,
                        evidence_tokens=counter.count(old_candidate)
                        - counter.count(old_base),
                        historical_evidence_tokens=historical_evidence_tokens,
                        original_completed_count=len(segments.completed),
                        projected_tool_message_count=projected_count,
                        tool_content_chars_removed=removed_chars,
                    )
                    if candidate_report.estimated_input_tokens <= budget.input_limit_tokens:
                        selected = old_candidate
                        budget_report = candidate_report
                        break
                if budget_report.estimated_input_tokens <= budget.input_limit_tokens:
                    break
            shorter = _drop_oldest_completed_turn(selected)
            if shorter is not None:
                selected = shorter
                continue
            if summary_text is not None:
                summary_text = None
                continue
            if historical_evidence_limit > 0 and historical_records:
                historical_evidence_limit = (
                    0
                    if historical_evidence_limit == 1
                    else max(1, historical_evidence_limit // 2)
                )
                continue
            for (
                candidate_base,
                candidate,
                projected_count,
                removed_chars,
                excerpt_chars_removed,
                records_omitted,
            ) in _projected_candidates(
                repaired,
                evidence_records,
                state.turn_evidence_omitted_count,
            ):
                budget_report = _budget_report(
                    selected,
                    candidate,
                    counter=counter,
                    budget=budget,
                    system_tokens=system_tokens,
                    directive_tokens=directive_tokens,
                    tool_schema_tokens=tool_schema_tokens,
                    summary_tokens=summary_tokens,
                    evidence_tokens=counter.count(candidate)
                    - counter.count(candidate_base),
                    historical_evidence_tokens=historical_evidence_tokens,
                    original_completed_count=len(segments.completed),
                    projected_tool_message_count=projected_count,
                    tool_content_chars_removed=removed_chars,
                    evidence_excerpt_chars_removed=excerpt_chars_removed,
                    evidence_records_omitted_by_budget=records_omitted,
                )
                if budget_report.estimated_input_tokens <= budget.input_limit_tokens:
                    selected = candidate
                    break
            else:
                reason = "CURRENT_TURN_CONTEXT_TOO_LARGE"
            if budget_report.estimated_input_tokens <= budget.input_limit_tokens:
                break
        raise ContextBudgetExceeded(
            reason,
            replace(budget_report, outcome=reason.lower()),
            completed_model_messages=tuple(latest_turn_ai_messages(source_messages)),
            compaction_usage=state.turn_compaction_usage,
        )

    if budget_report is not None and summary is not None and summary_text is None:
        budget_report = replace(
            budget_report,
            summary_omitted_by_budget=True,
            projection_reasons=budget_report.projection_reasons + ("summary_budget",),
        )
    if budget_report is not None and historical_evidence_omitted_by_budget:
        budget_report = replace(
            budget_report,
            evidence_records_omitted_by_budget=(
                budget_report.evidence_records_omitted_by_budget
                + historical_evidence_omitted_by_budget
            ),
            projection_reasons=budget_report.projection_reasons
            + ("historical_evidence_budget",),
        )
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
            + historical_evidence_omitted_by_budget
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
