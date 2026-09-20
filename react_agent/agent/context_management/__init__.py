"""Agent 模型上下文的分段、预算、协议修复、证据与压缩契约。"""

from react_agent.agent.context_management.builder import build_model_context
from react_agent.agent.context_management.contracts import (
    BudgetReport,
    ContextBudget,
    ContextBudgetExceeded,
    ContextDiagnostics,
    ConversationSummary,
    ModelContext,
    TurnSegments,
)
from react_agent.agent.context_management.segmentation import latest_turn_ai_messages


__all__ = [
    "BudgetReport",
    "ContextBudget",
    "ContextBudgetExceeded",
    "ContextDiagnostics",
    "ConversationSummary",
    "ModelContext",
    "TurnSegments",
    "build_model_context",
    "latest_turn_ai_messages",
]
