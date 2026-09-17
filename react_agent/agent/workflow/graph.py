"""Agent 执行图的定义与编译。

正常终止由显式业务预算控制：工具预算耗尽时先闭合未执行的调用，再由
不绑定工具的 finalize_model 生成最终回答。LangGraph recursion_limit 仅作为
异常循环熔断器，不参与正常业务路由。
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from langgraph.checkpoint.base import BaseCheckpointSaver

from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import InputState, State
from react_agent.agent.workflow.nodes import (
    call_model,
    close_pending_tool_calls,
    dynamic_tool_node,
    finalize_model,
    postprocess_tools,
    prepare_turn,
    reflection_node,
)
from react_agent.agent.workflow.routing import (
    route_after_postprocess,
    route_model_output,
)


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
