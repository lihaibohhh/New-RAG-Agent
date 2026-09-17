"""单次 Agent 编排所需的行为配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from react_agent.agent.configuration.environment import apply_environment_overrides
from react_agent.agent.policies.execution import (
    minimum_recursion_limit,
    validate_execution_budget,
)
from react_agent.agent.prompting.system import SYSTEM_PROMPT


@dataclass(kw_only=True)
class AgentContext:
    """Agent 编排上下文。

    这里只管理会影响图内决策的行为参数。模型选择与推理参数归
    ``core.LLMConfig``，具体工具参数归各工具配置，会话标识与持久化参数归
    ``conversations``。字段允许通过同名大写环境变量覆盖默认值。
    """

    # -----------------------------
    # Prompt / 语言与地域
    # -----------------------------

    system_prompt: str = field(
        default=SYSTEM_PROMPT,
        metadata={
            "description": "系统提示词（中文场景默认）。支持 {system_time} 变量注入。"
        },
    )

    language: str = field(
        default="zh-CN",
        metadata={"description": "默认语言。中文场景建议保持 zh-CN。"},
    )

    timezone: str = field(
        default="Asia/Shanghai",
        metadata={
            "description": "默认时区（IANA）。用于展示 system_time、时间相关回答等。"
        },
    )

    # -----------------------------
    # Agent 能力开关
    # -----------------------------
    enable_tools: bool = field(
        default=True,
        metadata={
            "description": "是否允许使用任何工具（总开关）。关闭后所有工具不可用"
        },
    )

    enable_web_search: bool = field(
        default=True,
        metadata={"description": "是否允许使用 web 搜索工具（Tavily）。"},
    )

    max_tool_output_chars: int = field(
        default=4000,
        metadata={"description": "工具 observation 的硬截断字符数（避免上下文污染）。"},
    )

    # -----------------------------
    # 图运行与安全兜底
    # -----------------------------

    recursion_limit: int = field(
        default=50,
        metadata={
            "description": "LangGraph 图执行熔断上限；不得用作 Agent 正常业务轮次预算。"
        },
    )

    max_model_rounds: int = field(
        default=8,
        metadata={
            "description": "单轮对话允许的常规 Agent 模型调用次数；最终总结调用单独预留。"
        },
    )

    max_tool_batches: int = field(
        default=6,
        metadata={"description": "单轮对话允许执行的工具批次数。"},
    )

    max_tool_retries: int = field(
        default=2,
        metadata={"description": "单轮对话中工具失败后的最大恢复重试次数。"},
    )

    rag_call_limit: int = field(
        default=3,
        metadata={
            "description": "单轮对话内 query_internal_knowledge 调用次数硬上限（兜底防御，与上层提示词策略无关）。"
        },
    )

    consecutive_failure_threshold: int = field(
        default=2,
        metadata={
            "description": "RAG 连续无效检索（has_relevant_content=False）次数达到该值后强制拒答终止。"
            "该值会注入 system_prompt 的拒答规则文案，修改此项无需再手动同步提示词。"
        },
    )

    # -----------------------------
    # 上下文与历史消息截断策略
    # -----------------------------

    max_history_tokens: int = field(
        default=64000,
        metadata={
            "description": "发给模型前保留的最大历史 token 数（不含本轮 system prompt）。"
            "按模型窗口 40%~60% 取值，给 completion 留足空间。"
        },
    )

    enable_history_truncation: bool = field(
        default=True,
        metadata={"description": "是否启用历史消息截断/压缩策略。"},
    )

    def __post_init__(self) -> None:
        apply_environment_overrides(self)
        self._validate_execution_budget()

    @property
    def minimum_recursion_limit(self) -> int:
        """返回当前业务预算在最坏节点路径下需要的图执行安全下限。"""
        return minimum_recursion_limit(
            max_model_rounds=self.max_model_rounds,
            max_tool_batches=self.max_tool_batches,
        )

    def _validate_execution_budget(self) -> None:
        validate_execution_budget(self)


__all__ = ["AgentContext"]
