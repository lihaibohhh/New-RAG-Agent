"""Agent 业务预算与主动终止策略。"""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class ExecutionBudget(Protocol):
    """预算策略需要的最小只读配置契约。"""

    recursion_limit: int
    max_model_rounds: int
    max_tool_batches: int
    max_tool_retries: int
    rag_call_limit: int
    consecutive_failure_threshold: int


class TerminationReason(str, Enum):
    MODEL_ROUND_BUDGET_EXHAUSTED = "MODEL_ROUND_BUDGET_EXHAUSTED"
    TOOL_BATCH_BUDGET_EXHAUSTED = "TOOL_BATCH_BUDGET_EXHAUSTED"
    TOOL_RETRY_BUDGET_EXHAUSTED = "TOOL_RETRY_BUDGET_EXHAUSTED"
    RAG_CALL_BUDGET_EXHAUSTED = "RAG_CALL_BUDGET_EXHAUSTED"
    RAG_CONSECUTIVE_MISS = "RAG_CONSECUTIVE_MISS"
    EVIDENCE_OUTPUT_BUDGET_EXHAUSTED = "EVIDENCE_OUTPUT_BUDGET_EXHAUSTED"
    GRAPH_STEP_BUDGET_EXHAUSTED = "GRAPH_STEP_BUDGET_EXHAUSTED"
    UNEXPECTED_RECURSION_BOUNDARY = "UNEXPECTED_RECURSION_BOUNDARY"


def allow_tool_path(remaining_steps: int) -> bool:
    """为工具执行、结果处理及最坏情况下的最终总结各留一步。"""
    return remaining_steps >= 4


def can_execute_tools(remaining_steps: int) -> bool:
    """工具节点之后还需结果处理与最终总结两个节点。"""
    return remaining_steps >= 3


def finalize_after_tools(remaining_steps: int) -> bool:
    """即将无法完成下一轮工具循环时，进入收口路径。"""
    return remaining_steps <= 3


def can_schedule_finalizer(remaining_steps: int) -> bool:
    """当前节点结束后是否还容得下独立的最终总结节点。"""
    return remaining_steps >= 2


def minimum_recursion_limit(
    *,
    max_model_rounds: int,
    max_tool_batches: int,
) -> int:
    """估算当前图拓扑下业务预算需要的递归安全下限。"""
    prepare_turn_steps = 1
    model_steps = max_model_rounds
    tool_path_steps = 3 * max_tool_batches
    close_and_finalize_steps = 2
    safety_margin = 4
    return (
        prepare_turn_steps
        + model_steps
        + tool_path_steps
        + close_and_finalize_steps
        + safety_margin
    )


def validate_execution_budget(config: ExecutionBudget) -> None:
    """校验业务预算为正，且图熔断上限能够覆盖正常业务路径。"""
    limits = {
        "max_model_rounds": config.max_model_rounds,
        "max_tool_batches": config.max_tool_batches,
        "max_tool_retries": config.max_tool_retries,
    }
    invalid = [name for name, value in limits.items() if value < 1]
    if invalid:
        raise ValueError(f"Agent 业务预算必须大于 0: {', '.join(invalid)}")

    required = minimum_recursion_limit(
        max_model_rounds=config.max_model_rounds,
        max_tool_batches=config.max_tool_batches,
    )
    if config.recursion_limit < required:
        raise ValueError(
            "recursion_limit 不足以覆盖当前 Agent 业务预算："
            f"配置值={config.recursion_limit}，至少需要={required}。"
            "请提高 recursion_limit 或降低 max_model_rounds/max_tool_batches。"
        )


def model_termination_reason(
    config: ExecutionBudget,
    *,
    tool_names: list[str],
    next_model_round: int,
    completed_tool_batches: int,
    attempted_rag_calls: int,
) -> str | None:
    """模型请求工具后，判断是否应先闭合调用并主动收口。"""
    if not tool_names:
        return None
    if (
        attempted_rag_calls >= config.rag_call_limit
        and "query_internal_knowledge" in tool_names
    ):
        return TerminationReason.RAG_CALL_BUDGET_EXHAUSTED.value
    if next_model_round >= config.max_model_rounds:
        return TerminationReason.MODEL_ROUND_BUDGET_EXHAUSTED.value
    if completed_tool_batches >= config.max_tool_batches:
        return TerminationReason.TOOL_BATCH_BUDGET_EXHAUSTED.value
    return None


def tool_batch_termination_reason(
    config: ExecutionBudget,
    *,
    consecutive_rag_misses: int,
    next_tool_batch: int,
    has_errors: bool,
    completed_retries: int,
) -> str | None:
    """工具结果归一化后，判断是否应进入无工具最终总结。"""
    if consecutive_rag_misses >= config.consecutive_failure_threshold:
        return TerminationReason.RAG_CONSECUTIVE_MISS.value
    if next_tool_batch >= config.max_tool_batches:
        return TerminationReason.TOOL_BATCH_BUDGET_EXHAUSTED.value
    if has_errors and completed_retries >= config.max_tool_retries:
        return TerminationReason.TOOL_RETRY_BUDGET_EXHAUSTED.value
    return None


__all__ = [
    "ExecutionBudget",
    "TerminationReason",
    "allow_tool_path",
    "can_execute_tools",
    "can_schedule_finalizer",
    "finalize_after_tools",
    "minimum_recursion_limit",
    "model_termination_reason",
    "tool_batch_termination_reason",
    "validate_execution_budget",
]
