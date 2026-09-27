"""Agent 图运行所需能力的显式依赖契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from react_agent.agent.config import AgentContext
from react_agent.metering.contracts import CostEstimator
from react_agent.skills import SkillRegistry


ModelProvider = Callable[[], BaseChatModel]
OutputTokenLimiter = Callable[..., Any]


class ToolCapability(str, Enum):
    """Agent 用于制定策略的稳定工具能力，而非具体适配器名称。"""

    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    WEB_SEARCH = "web_search"


@dataclass(frozen=True)
class AgentDependencies:
    """由 Composition Root 提供、由 Agent 图统一消费的运行能力。

    本对象只管理 Agent 执行所需的稳定能力，不负责读取环境变量、选择具体
    基础设施、创建 RAG Runtime 或管理资源生命周期。
    """

    config: AgentContext
    model_provider: ModelProvider
    tools: tuple[BaseTool, ...]
    tool_capabilities: Mapping[str, frozenset[ToolCapability]] | None = None
    skill_registry: SkillRegistry | None = None
    model_ref: str = "unknown"
    cost_estimator: CostEstimator | None = None
    output_token_limiter: OutputTokenLimiter | None = None
    model_context_window_tokens: int | None = None
    reserved_completion_tokens: int = 8192

    def __post_init__(self) -> None:
        if not isinstance(self.config, AgentContext):
            raise TypeError("config 必须是 AgentContext")
        if not callable(self.model_provider):
            raise TypeError("model_provider 必须可调用")
        if not self.model_ref.strip():
            raise ValueError("model_ref 不能为空")
        if self.cost_estimator is not None and not callable(self.cost_estimator):
            raise TypeError("cost_estimator 必须可调用")
        if self.output_token_limiter is not None and not callable(
            self.output_token_limiter
        ):
            raise TypeError("output_token_limiter 必须可调用")
        if self.model_context_window_tokens is not None and self.model_context_window_tokens < 1:
            raise ValueError("model_context_window_tokens 必须大于 0")
        if self.reserved_completion_tokens < 1:
            raise ValueError("reserved_completion_tokens 必须大于 0")

        tools = tuple(self.tools)
        names: list[str] = []
        for agent_tool in tools:
            name = str(getattr(agent_tool, "name", "") or "").strip()
            if not name:
                raise ValueError("Agent 工具必须提供非空 name")
            names.append(name)
        duplicate_names = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicate_names:
            raise ValueError("Agent 工具名称不能重复: " + ", ".join(duplicate_names))
        object.__setattr__(self, "tools", tools)
        capabilities: dict[str, frozenset[ToolCapability]] = {}
        for tool_name, raw_capabilities in (self.tool_capabilities or {}).items():
            if tool_name not in names:
                raise ValueError(f"工具能力映射引用了未注册工具: {tool_name}")
            capabilities[tool_name] = frozenset(
                ToolCapability(capability) for capability in raw_capabilities
            )
        object.__setattr__(
            self,
            "tool_capabilities",
            MappingProxyType(capabilities),
        )

    def resolve_model(self) -> BaseChatModel:
        """按需取得由 Runtime 选择的聊天模型。"""
        return self.model_provider()

    def limit_model_output(self, model: BaseChatModel, max_tokens: int) -> Any:
        """Apply the Runtime-injected provider-specific output limit."""
        if self.output_token_limiter is not None:
            return self.output_token_limiter(model, max_tokens=max_tokens)
        bind = getattr(model, "bind", None)
        return bind(max_tokens=max_tokens) if callable(bind) else model

    def active_tools(self) -> tuple[tuple[BaseTool, ...], frozenset[str]]:
        """返回由 Runtime 注入且受 Agent 总开关控制的工具。"""
        if not self.config.enable_tools:
            return (), frozenset()
        active = self.tools
        return active, frozenset(agent_tool.name for agent_tool in active)

    def tool_names_for(self, capability: ToolCapability) -> frozenset[str]:
        """返回具备指定能力的已注册工具名称。"""
        return frozenset(
            tool_name
            for tool_name, capabilities in self.tool_capabilities.items()
            if capability in capabilities
        )

__all__ = [
    "AgentDependencies",
    "ModelProvider",
    "OutputTokenLimiter",
    "ToolCapability",
]
