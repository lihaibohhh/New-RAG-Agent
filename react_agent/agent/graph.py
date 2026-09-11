"""Agent 执行图的定义与编译。

核心改进：
1. 解决 AsyncSqliteSaver 的异步初始化问题
2. 支持动态 checkpointer 创建（延迟初始化）
3. 保持向后兼容，默认使用 MemorySaver
4. 提供工程级别的错误处理和降级策略
5. 集成 Postgres 持久化
6. ✅ 新增 Reflection Node（反思节点）：工具报错时自动介入
7. 优化图结构：Tools -> Reflection -> Model
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from langgraph.checkpoint.base import BaseCheckpointSaver

from react_agent.agent.dependencies import AgentDependencies
from react_agent.agent.state import InputState, State
from react_agent.agent.routing import route_model_output, route_after_postprocess
from react_agent.agent.nodes import call_model, dynamic_tool_node, postprocess_tools, reflection_node


# ==================== 图构建器（基础版）====================
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

    builder.add_node("call_model", call_model)
    builder.add_node("tools", dynamic_tool_node)
    builder.add_node("postprocess_tools", postprocess_tools)
    builder.add_node("reflection", reflection_node)  # ✅ 新增：注册反思节点

    builder.add_edge("__start__", "call_model")
    builder.add_conditional_edges("call_model", route_model_output)
    builder.add_edge("tools", "postprocess_tools")

    # Postprocess -> Reflection (✅ 关键改变：处理完结果后，先去反思节点检查有没有错)
    builder.add_conditional_edges("postprocess_tools", route_after_postprocess)

    # Reflection -> Call Model (✅ 闭环：反思完（无论有错没错），都回模型继续思考)
    builder.add_edge("reflection", "call_model")

    return builder


# Checkpointer 由 Composition Root 创建并注入，Agent 模块
# 不负责选择数据库、创建连接池或管理持久化资源生命周期。
def compile_agent_graph(checkpointer: BaseCheckpointSaver | None = None):
    """使用调用方提供的 Checkpointer 编译 Agent 图。"""
    builder = build_base_graph()

    return builder.compile(
        name="ReAct Agent (ZH) - Persistent",
        checkpointer=checkpointer
    )


