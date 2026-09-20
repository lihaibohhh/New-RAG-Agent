"""上下文构造与后续压缩使用的稳定数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TurnSegments:
    """已完成历史与当前用户轮次的消息切片。"""

    completed: tuple[Any, ...] = ()
    current: tuple[Any, ...] = ()


@dataclass(frozen=True)
class TurnBlock:
    """一组连续的用户输入及其后续模型和工具消息。"""

    messages: tuple[Any, ...]


@dataclass(frozen=True)
class ToolExchange:
    """一次模型工具调用及其按 ID 配对的全部工具结果。"""

    assistant_message: Any
    tool_messages: tuple[Any, ...]
    call_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConversationSummary:
    """较早已完成轮次的可追溯压缩结果。"""

    content: str
    source_message_ids: tuple[str, ...] = ()
    summarized_through_message_id: str | None = None
    version: int = 1
    exact_excerpts: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    source_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("ConversationSummary.content 不能为空")
        if self.version < 1:
            raise ValueError("ConversationSummary.version 必须大于 0")
        if self.source_message_ids and (
            self.summarized_through_message_id != self.source_message_ids[-1]
        ):
            raise ValueError("摘要游标必须是最后一条来源消息 ID")


@dataclass(frozen=True)
class ContextBudget:
    """一次模型调用的输入策略上限与可选模型窗口能力。"""

    max_input_tokens: int
    reserved_completion_tokens: int
    safety_margin_tokens: int
    context_window_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens 必须大于 0")
        if self.reserved_completion_tokens < 1:
            raise ValueError("reserved_completion_tokens 必须大于 0")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens 不能为负数")
        if self.context_window_tokens is not None and self.context_window_tokens < 1:
            raise ValueError("context_window_tokens 必须大于 0")

    @property
    def input_limit_tokens(self) -> int:
        """未配置模型窗口时只执行本地策略上限，不宣称 Provider 安全。"""
        if self.context_window_tokens is None:
            return self.max_input_tokens
        return min(
            self.max_input_tokens,
            self.context_window_tokens
            - self.reserved_completion_tokens
            - self.safety_margin_tokens,
        )


@dataclass(frozen=True)
class BudgetReport:
    """本次实际模型输入的分项估算。"""

    input_limit_tokens: int
    estimated_input_tokens: int
    system_tokens: int
    directive_tokens: int
    tool_schema_tokens: int
    current_user_tokens: int
    current_turn_other_tokens: int
    evidence_tokens: int
    recent_history_tokens: int
    reserved_completion_tokens: int
    safety_margin_tokens: int
    context_window_tokens: int | None
    summary_tokens: int = 0
    historical_evidence_tokens: int = 0
    dropped_completed_message_count: int = 0
    outcome: str = "within_budget"
    projected_tool_message_count: int = 0
    tool_content_chars_removed: int = 0
    evidence_excerpt_chars_removed: int = 0
    evidence_records_omitted_by_budget: int = 0
    projection_reasons: tuple[str, ...] = ()
    summary_omitted_by_budget: bool = False


class ContextBudgetExceeded(ValueError):
    """调用模型前发现本次输入无法在预算内安全构造。"""

    def __init__(
        self,
        reason: str,
        report: BudgetReport,
        *,
        completed_model_messages: tuple[Any, ...] = (),
        compaction_usage: dict[str, Any] | None = None,
    ) -> None:
        self.reason = reason
        self.report = report
        self.completed_model_messages = completed_model_messages
        self.compaction_usage = compaction_usage
        super().__init__(reason)


@dataclass(frozen=True)
class ContextDiagnostics:
    """不进入提示词的上下文构造诊断数据。"""

    source_message_count: int
    selected_message_count: int
    estimated_history_tokens: int
    completed_message_count: int = 0
    current_turn_message_count: int = 0
    evidence_record_count: int = 0
    evidence_omitted_count: int = 0
    historical_evidence_record_count: int = 0
    historical_evidence_omitted_count: int = 0
    history_truncated: bool = False
    summary_included: bool = False
    summarized_message_count: int = 0


@dataclass(frozen=True)
class ModelContext:
    """一次模型调用的完整输入和可观测诊断。"""

    messages: tuple[Any, ...]
    diagnostics: ContextDiagnostics
    budget_report: BudgetReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_list(self) -> list[Any]:
        return list(self.messages)


__all__ = [
    "BudgetReport",
    "ContextBudget",
    "ContextBudgetExceeded",
    "ContextDiagnostics",
    "ConversationSummary",
    "ModelContext",
    "ToolExchange",
    "TurnBlock",
    "TurnSegments",
]
