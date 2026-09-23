"""为一次模型调用选择可放入上下文窗口的完整消息。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from react_agent.agent.context_management.budgeting import (
    MessageTokenCounter,
    estimate_tool_schema_tokens,
)
from react_agent.agent.context_management.contracts import (
    BudgetReport,
    ContextBudget,
    ContextBudgetExceeded,
)
from react_agent.agent.context_management.evidence import (
    attach_evidence_index,
    render_evidence_index,
    render_historical_evidence_index,
    select_historical_evidence,
)
from react_agent.agent.context_management.projection import (
    project_completed_tool_messages,
    project_current_tool_messages,
)
from react_agent.agent.context_management.protocol import sanitize_dangling_tool_calls
from react_agent.agent.context_management.segmentation import (
    segment_current_turn,
    segment_turn_blocks,
)


@dataclass(frozen=True)
class ContextSelection:
    """预算策略选出的模型可见消息及随之变化的输入片段。"""

    messages: list[Any]
    summary_text: str | None
    historical_evidence_text: str
    historical_evidence_omitted_by_budget: int
    budget_report: BudgetReport | None


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
    specs = [(None, excerpt_chars, None) for excerpt_chars in (120, 60, 0)]
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


def select_context_for_budget(
    selected: list[Any],
    *,
    counter: MessageTokenCounter,
    budget: ContextBudget | None,
    system_prompt: str,
    directive: str | None,
    bound_tools: tuple[Any, ...],
    summary_text: str | None,
    original_completed_count: int,
    turn_evidence: list[dict[str, Any]],
    turn_evidence_omitted_count: int,
    conversation_evidence: list[dict[str, Any]],
    conversation_evidence_omitted_count: int,
) -> ContextSelection:
    """按固定的降级顺序选择上下文，保持当前轮的工具交换完整。"""
    had_summary = summary_text is not None
    evidence_index = render_evidence_index(
        turn_evidence, omitted_count=turn_evidence_omitted_count
    )
    historical_evidence_limit = len(conversation_evidence)
    historical_evidence_text = ""
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
        historical_records = select_historical_evidence(
            conversation_evidence, repaired
        )
        if historical_evidence_limit > 0 and historical_records:
            historical_evidence_text = render_historical_evidence_index(
                historical_records,
                omitted_count=conversation_evidence_omitted_count,
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
            counter.count([{"role": "assistant", "content": historical_evidence_text}])
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
            original_completed_count=original_completed_count,
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
                        original_completed_count=original_completed_count,
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
                repaired, turn_evidence, turn_evidence_omitted_count
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
                    original_completed_count=original_completed_count,
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
            reason, replace(budget_report, outcome=reason.lower())
        )

    if budget_report is not None and had_summary and summary_text is None:
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
    return ContextSelection(
        messages=selected,
        summary_text=summary_text,
        historical_evidence_text=historical_evidence_text,
        historical_evidence_omitted_by_budget=historical_evidence_omitted_by_budget,
        budget_report=budget_report,
    )
