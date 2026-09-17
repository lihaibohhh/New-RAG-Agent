from __future__ import annotations
import logging
from typing import Literal

from langchain_core.messages import AIMessage
from react_agent.agent.contracts.state import State
from react_agent.agent.tool_flow.calls import extract_tool_call_ids


logger = logging.getLogger(__name__)


def route_model_output(
    state: State,
) -> Literal["__end__", "tools", "call_model", "close_pending_tool_calls"]:
    """根据最后一条 AIMessage 是否包含 tool_calls 决定下一步"""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        # FIX-⑦: 非 AIMessage 时不崩溃，回退到 call_model 重新生成，并记录日志便于排查
        logger.warning(
            "[route_model_output] 最后一条消息不是 AIMessage，实际类型: %s，回退到 call_model",
            type(last_message).__name__,
        )
        return "call_model"
    if not extract_tool_call_ids(last_message):
        return "__end__"
    if state.termination_reason:
        return "close_pending_tool_calls"
    return "tools"


def route_after_postprocess(
    state: State,
) -> Literal["call_model", "reflection", "finalize_model"]:
    """按显式业务终止原因和当前工具批次状态选择后续节点。"""
    if state.termination_reason:
        return "finalize_model"
    if state.last_tool_batch_errors:
        return "reflection"
    return "call_model"
