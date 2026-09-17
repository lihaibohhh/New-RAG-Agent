"""Agent 图运行所需能力的显式依赖契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from react_agent.agent.configuration.context import AgentContext
from react_agent.metering.contracts import CostEstimator


ModelProvider = Callable[[], BaseChatModel]


@dataclass(frozen=True)
class AgentDependencies:
    """由 Composition Root 提供、由 Agent 图统一消费的运行能力。

    本对象只管理 Agent 执行所需的稳定能力，不负责读取环境变量、选择具体
    基础设施、创建 RAG Runtime 或管理资源生命周期。
    """

    config: AgentContext
    model_provider: ModelProvider
    tools: tuple[BaseTool, ...]
    model_ref: str = "unknown"
    cost_estimator: CostEstimator | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, AgentContext):
            raise TypeError("config 必须是 AgentContext")
        if not callable(self.model_provider):
            raise TypeError("model_provider 必须可调用")
        if not self.model_ref.strip():
            raise ValueError("model_ref 不能为空")
        if self.cost_estimator is not None and not callable(self.cost_estimator):
            raise TypeError("cost_estimator 必须可调用")

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

    def resolve_model(self) -> BaseChatModel:
        """按需取得由 Runtime 选择的聊天模型。"""
        return self.model_provider()

    def active_tools(self) -> tuple[tuple[BaseTool, ...], frozenset[str]]:
        """根据 Agent 配置返回本轮允许绑定和执行的工具。"""
        if not self.config.enable_tools:
            return (), frozenset()
        active = tuple(
            agent_tool
            for agent_tool in self.tools
            if agent_tool.name != "search" or self.config.enable_web_search
        )
        return active, frozenset(agent_tool.name for agent_tool in active)


__all__ = ["AgentDependencies", "ModelProvider"]
