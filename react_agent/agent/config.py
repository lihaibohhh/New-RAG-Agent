"""单次 Agent 编排所需的行为配置。"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from react_agent.agent.policies import (
    minimum_recursion_limit,
    validate_execution_budget,
)
from react_agent.agent.prompts import SYSTEM_PROMPT


logger = logging.getLogger(__name__)


def _to_bool(value: str) -> bool:
    return (value or "").strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }


def _unwrap_annotated(field_type: Any) -> Any:
    if get_origin(field_type) is Annotated:
        return get_args(field_type)[0]
    return field_type


def _coerce(value: str, field_type: Any) -> Any:
    """将环境变量字符串转换成 dataclass 字段声明的类型。"""
    field_type = _unwrap_annotated(field_type)
    origin = get_origin(field_type)

    if origin is None:
        if field_type is bool:
            return _to_bool(value)
        if field_type is int:
            return int(value)
        if field_type is float:
            return float(value)
        return value

    if origin is list:
        inner = get_args(field_type)[0] if get_args(field_type) else str
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [_coerce(part, inner) for part in parts]

    if origin in {dict, tuple}:
        return value

    if origin in {Union, UnionType}:
        for candidate in get_args(field_type):
            if candidate is type(None):
                continue
            try:
                return _coerce(value, candidate)
            except (TypeError, ValueError):
                continue

    return value


def apply_environment_overrides(target: Any) -> None:
    """以大写字段名为 key，将环境变量覆盖到仍使用默认值的字段。"""
    type_hints = get_type_hints(target.__class__, include_extras=True)
    for data_field in fields(target):
        if not data_field.init:
            continue

        current = getattr(target, data_field.name)
        if current != data_field.default:
            continue

        env_key = data_field.name.upper()
        raw = os.environ.get(env_key)
        if raw is None or not raw.strip():
            continue

        field_type = type_hints.get(data_field.name, data_field.type)
        try:
            setattr(target, data_field.name, _coerce(raw, field_type))
        except (TypeError, ValueError):
            logger.warning(
                "[AgentContext] env %s=%r cannot be converted to %s; ignored",
                env_key,
                raw,
                field_type,
            )


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
        default=6,
        metadata={
            "description": (
                "单轮对话内被 Agent 接受并执行的 "
                "query_internal_knowledge 调用次数硬上限；"
                "成功与失败均占用配额，本地预检拦截不占用。"
            )
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
        default=80000,
        metadata={
            "description": "发给模型前保留的最大历史估算 token 数；最终仍受总输入预算约束。"
        },
    )

    max_input_tokens: int = field(
        default=96000,
        metadata={
            "description": "单次模型调用的本地输入估算上限；包含提示词、工具 schema、证据和历史。"
        },
    )

    context_safety_margin_tokens: int = field(
        default=1024,
        metadata={"description": "已配置模型上下文窗口时，为估算误差预留的 token 数。"},
    )

    enable_history_truncation: bool = field(
        default=True,
        metadata={"description": "是否启用历史消息截断；不影响总输入预算闸门。"},
    )

    enable_history_compaction: bool = field(
        default=True,
        metadata={"description": "超出历史触发阈值时，压缩较早已完成轮次。"},
    )

    history_compaction_max_output_tokens: int = field(
        default=768,
        metadata={"description": "历史摘要模型单次最大输出 Token。"},
    )

    history_compaction_retry_new_tokens: int = field(
        default=512,
        metadata={
            "description": "压缩被拒绝后，候选来源至少新增该 Token 数才重试。"
        },
    )

    def __post_init__(self) -> None:
        apply_environment_overrides(self)
        self._validate_execution_budget()
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens 必须大于 0")
        if self.context_safety_margin_tokens < 0:
            raise ValueError("context_safety_margin_tokens 不能为负数")
        if self.history_compaction_max_output_tokens < 1:
            raise ValueError("history_compaction_max_output_tokens 必须大于 0")
        if self.history_compaction_retry_new_tokens < 1:
            raise ValueError("history_compaction_retry_new_tokens 必须大于 0")

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
