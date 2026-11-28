"""Agent 执行图的定义与编译。

正常终止由显式业务预算及 LangGraph 剩余步骤共同控制：工具预算耗尽时先
闭合未执行的调用，再由不绑定工具的 finalize_model 生成最终回答。
recursion_limit 本身仍是异常循环熔断器。
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import InputState, State
from react_agent.agent.policies import TerminationReason
from react_agent.agent.tool_flow import extract_tool_call_ids
from react_agent.agent.workflow.nodes import (
    call_model,
    close_pending_tool_calls,
    dynamic_tool_node,
    finalize_model,
    postprocess_tools,
    prepare_turn,
    reflection_node,
)


logger = logging.getLogger(__name__)


def route_model_output(
    state: State,
) -> Literal["__end__", "tools", "call_model", "close_pending_tool_calls"]:
    """根据最后一条模型消息和终止状态选择下一节点。"""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        logger.warning(
            "[route_model_output] 最后一条消息不是 AIMessage，实际类型: %s，"
            "回退到 call_model",
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
) -> Literal["__end__", "call_model", "reflection", "finalize_model"]:
    """按终止原因和工具批次状态选择后续节点。"""
    if (
        state.termination_reason
        == TerminationReason.UNEXPECTED_RECURSION_BOUNDARY.value
    ):
        return "__end__"
    if state.termination_reason:
        return "finalize_model"
    if state.last_tool_batch_errors:
        return "reflection"
    return "call_model"


def build_base_graph() -> StateGraph:
    """
    构建基础图（不包含 checkpointer）

    这样可以：
    1. 预先定义图结构
    2. 在运行时动态添加 checkpointer
    3. 支持多种编译模式
    """
    builder = StateGraph(
        State,
        input_schema=InputState,
        context_schema=AgentDependencies,
    )

    builder.add_node("prepare_turn", prepare_turn)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", dynamic_tool_node)
    builder.add_node("postprocess_tools", postprocess_tools)
    builder.add_node("reflection", reflection_node)
    builder.add_node("close_pending_tool_calls", close_pending_tool_calls)
    builder.add_node("finalize_model", finalize_model)

    builder.add_edge("__start__", "prepare_turn")
    builder.add_edge("prepare_turn", "call_model")
    builder.add_conditional_edges("call_model", route_model_output)
    builder.add_edge("tools", "postprocess_tools")
    builder.add_conditional_edges("postprocess_tools", route_after_postprocess)
    builder.add_edge("reflection", "call_model")
    builder.add_edge("close_pending_tool_calls", "finalize_model")
    builder.add_edge("finalize_model", "__end__")

    return builder


# Checkpointer 由 Composition Root 创建并注入，Agent 模块
# 不负责选择数据库、创建连接池或管理持久化资源生命周期。
def compile_agent_graph(checkpointer: BaseCheckpointSaver | None = None):
    """使用调用方提供的 Checkpointer 编译 Agent 图。"""
    builder = build_base_graph()

    return builder.compile(
        name="ReAct Agent (ZH) - Persistent", checkpointer=checkpointer
    )
