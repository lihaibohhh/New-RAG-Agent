"""上下文构造与后续压缩使用的稳定数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_COMPRESSION_LEVELS = frozenset({"L0", "L1", "L2", "L3"})


@dataclass(frozen=True)
class EvidenceSnapshot:
    """已完成轮次的无工具协议证据快照。

    ``exact_excerpt`` 只表示当时 ToolMessage 中模型可见的确定性摘录，
    不声称是底层文档的完整原文。
    """

    kind: str
    query: str
    exact_excerpt: str
    source_file: str = ""
    source_page: int | None = None
    chunk_id: str = ""
    url: str = ""
    title: str = ""
    published_at: str | None = None
    retrieved_at: str | None = None
    rank: int | None = None
    excerpt_truncated: bool = False
    tool_message_id: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"rag", "web_search", "other"}:
            raise ValueError(f"不支持的证据类型: {self.kind}")
        if not self.exact_excerpt.strip():
            raise ValueError("EvidenceSnapshot.exact_excerpt 不能为空")
        if self.source_page is not None and self.source_page < 1:
            raise ValueError("EvidenceSnapshot.source_page 必须大于 0")
        if self.rank is not None and self.rank < 1:
            raise ValueError("EvidenceSnapshot.rank 必须大于 0")

    def to_state(self) -> dict[str, Any]:
        """转换为 Checkpoint 可直接持久化的结构。"""
        return {
            "kind": self.kind,
            "query": self.query,
            "exact_excerpt": self.exact_excerpt,
            "source_file": self.source_file,
            "source_page": self.source_page,
            "chunk_id": self.chunk_id,
            "url": self.url,
            "title": self.title,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "rank": self.rank,
            "excerpt_truncated": self.excerpt_truncated,
            "tool_message_id": self.tool_message_id,
        }

    @classmethod
    def from_state(cls, raw: dict[str, Any]) -> "EvidenceSnapshot":
        """从持久化结构恢复并重新执行契约校验。"""
        return cls(
            kind=str(raw.get("kind") or ""),
            query=str(raw.get("query") or ""),
            exact_excerpt=str(raw.get("exact_excerpt") or ""),
            source_file=str(raw.get("source_file") or ""),
            source_page=raw.get("source_page"),
            chunk_id=str(raw.get("chunk_id") or ""),
            url=str(raw.get("url") or ""),
            title=str(raw.get("title") or ""),
            published_at=(
                str(raw["published_at"]) if raw.get("published_at") else None
            ),
            retrieved_at=(
                str(raw["retrieved_at"]) if raw.get("retrieved_at") else None
            ),
            rank=raw.get("rank"),
            excerpt_truncated=bool(raw.get("excerpt_truncated")),
            tool_message_id=str(raw.get("tool_message_id") or ""),
        )


@dataclass(frozen=True)
class TurnMemory:
    """从原始消息派生的、以用户轮次为边界的模型记忆。"""

    turn_id: str
    user_messages: tuple[str, ...]
    final_answer: str | None
    summary: str | None = None
    evidence: tuple[EvidenceSnapshot, ...] = ()
    compression_level: str = "L0"
    answer_omitted: bool = False
    evidence_omitted: bool = False
    evidence_omitted_count: int = 0
    source_message_ids: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if not self.turn_id.strip():
            raise ValueError("TurnMemory.turn_id 不能为空")
        if not self.user_messages or not any(
            message.strip() for message in self.user_messages
        ):
            raise ValueError("TurnMemory.user_messages 不能为空")
        if self.compression_level not in _COMPRESSION_LEVELS:
            raise ValueError(
                f"不支持的压缩等级: {self.compression_level}"
            )
        if self.evidence_omitted_count < 0:
            raise ValueError("TurnMemory.evidence_omitted_count 不能为负数")
        if self.version < 1:
            raise ValueError("TurnMemory.version 必须大于 0")

    def to_state(self) -> dict[str, Any]:
        """转换为 Checkpoint 可直接持久化的结构。"""
        return {
            "turn_id": self.turn_id,
            "user_messages": list(self.user_messages),
            "final_answer": self.final_answer,
            "summary": self.summary,
            "evidence": [record.to_state() for record in self.evidence],
            "compression_level": self.compression_level,
            "answer_omitted": self.answer_omitted,
            "evidence_omitted": self.evidence_omitted,
            "evidence_omitted_count": self.evidence_omitted_count,
            "source_message_ids": list(self.source_message_ids),
            "version": self.version,
        }

    @classmethod
    def from_state(cls, raw: dict[str, Any]) -> "TurnMemory":
        """从持久化结构恢复并重新执行契约校验。"""
        raw_evidence = raw.get("evidence")
        raw_evidence = raw_evidence if isinstance(raw_evidence, list) else []
        return cls(
            turn_id=str(raw.get("turn_id") or ""),
            user_messages=tuple(
                str(message) for message in raw.get("user_messages", ())
            ),
            final_answer=(
                str(raw["final_answer"])
                if raw.get("final_answer") is not None
                else None
            ),
            summary=(str(raw["summary"]) if raw.get("summary") else None),
            evidence=tuple(
                EvidenceSnapshot.from_state(record)
                for record in raw_evidence
                if isinstance(record, dict)
            ),
            compression_level=str(raw.get("compression_level") or "L0"),
            answer_omitted=bool(raw.get("answer_omitted")),
            evidence_omitted=bool(raw.get("evidence_omitted")),
            evidence_omitted_count=int(raw.get("evidence_omitted_count") or 0),
            source_message_ids=tuple(
                str(message_id) for message_id in raw.get("source_message_ids", ())
            ),
            version=int(raw.get("version") or 1),
        )


@dataclass(frozen=True)
class TurnSegments:
    """已完成历史与当前用户轮次的消息切片。"""

    completed: tuple[Any, ...] = ()
    current: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ContextBudget:
    """一次模型调用的输入策略上限与可选模型窗口能力。"""

    max_input_tokens: int
    reserved_completion_tokens: int
    safety_margin_tokens: int
    context_window_tokens: int | None = None
    target_input_tokens: int | None = None
    token_estimate_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens 必须大于 0")
        if self.reserved_completion_tokens < 1:
            raise ValueError("reserved_completion_tokens 必须大于 0")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens 不能为负数")
        if self.context_window_tokens is not None and self.context_window_tokens < 1:
            raise ValueError("context_window_tokens 必须大于 0")
        if self.target_input_tokens is not None and self.target_input_tokens < 1:
            raise ValueError("target_input_tokens 必须大于 0")
        if self.token_estimate_multiplier < 1.0:
            raise ValueError("token_estimate_multiplier 不能小于 1.0")

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

    @property
    def input_target_tokens(self) -> int:
        """软目标永不超过模型能力与本地硬上限。"""
        return min(self.target_input_tokens or self.input_limit_tokens, self.input_limit_tokens)


@dataclass(frozen=True)
class BudgetReport:
    """本次实际模型输入的分项估算。"""

    input_limit_tokens: int
    input_target_tokens: int
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
    historical_evidence_tokens: int = 0
    dropped_completed_message_count: int = 0
    outcome: str = "within_budget"
    projected_tool_message_count: int = 0
    tool_content_chars_removed: int = 0
    evidence_excerpt_chars_removed: int = 0
    evidence_records_omitted_by_budget: int = 0
    projection_reasons: tuple[str, ...] = ()
    historical_turn_levels: tuple[tuple[str, str], ...] = ()
    token_estimate_multiplier: float = 1.0
    soft_target_exceeded: bool = False


class ContextBudgetExceeded(ValueError):
    """调用模型前发现本次输入无法在预算内安全构造。"""

    def __init__(
        self,
        reason: str,
        report: BudgetReport,
        *,
        completed_model_messages: tuple[Any, ...] = (),
    ) -> None:
        self.reason = reason
        self.report = report
        self.completed_model_messages = completed_model_messages
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
    "EvidenceSnapshot",
    "ModelContext",
    "TurnMemory",
    "TurnSegments",
]
