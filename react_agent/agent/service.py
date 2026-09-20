"""Agent 对话用例服务。"""

from __future__ import annotations
import logging
from typing import Any, List
from react_agent.agent.config import AgentContext
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.metering.turn import extract_cumulative_snapshot

_logger = logging.getLogger(__name__)


class AgentService:
    """
    Agent 对话用例服务。

    只负责执行 Agent 用例。执行图和 Checkpointer 由 Composition Root
    创建并注入；会话历史查询、删除等管理操作属于 conversations 模块。
    """

    def __init__(self, dependencies: AgentDependencies, graph: Any):
        self._dependencies = dependencies
        self.ctx = dependencies.config
        self._graph = graph
        self._initialized = True

    @property
    def context(self) -> AgentContext:
        """Read-only access to the execution configuration needed by adapters."""
        return self.ctx

    @property
    def model_ref(self) -> str:
        """返回 Runtime 注入的模型引用，供入站适配器展示或兜底。"""
        return self._dependencies.model_ref

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """兼容旧调用；对象由 Composition Root 完整创建后才会被暴露。"""
        return None

    async def get_usage_snapshot(self, thread_id: str) -> dict[str, Any]:
        """读取持久化累计用量，供入站适配器建立本次 UI 会话的差值基线。"""
        if not thread_id:
            raise ValueError("thread_id 不能为空")
        snapshot = await self._graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        values = dict(getattr(snapshot, "values", None) or {})
        return extract_cumulative_snapshot(values)

    async def invoke(self, messages: List[Any], thread_id: str):
        """
        调用 agent

        Args:
            messages: 消息列表（只传当前这一条新消息即可）
            thread_id: 必填会话 ID，调用方负责用户命名空间隔离
        """
        if not self._initialized:
            await self.initialize()

        if not thread_id:
            _logger.warning("[Agent] invoke() 调用时未传 thread_id。")
            raise ValueError(
                "thread_id 不能为空。请传入 user:{username} 格式的 thread_id。"
            )

        config = {
            "recursion_limit": self.ctx.recursion_limit,
            "configurable": {"thread_id": thread_id},
        }

        result = await self._graph.ainvoke(
            {"messages": messages}, context=self._dependencies, config=config
        )

        return result

    async def stream_events(self, messages: List[Any], thread_id: str):
        """
        Token 级流式调用，使用 astream_events(version='v2')。

        Yields 原始 LangGraph 事件字典。调用方（路由层）负责将事件映射为 SSE 帧。
        call_model 节点须声明 config: RunnableConfig 并透传给 model.ainvoke()，
        否则回调链断裂，on_chat_model_stream 事件不会出现。
        """
        if not self._initialized:
            await self.initialize()

        if not thread_id:
            _logger.warning(
                "[Agent] stream_events() 调用时未传 thread_id，"
                "请传入 user:{username} 格式的 thread_id。"
            )
            raise ValueError(
                "thread_id 不能为空。请传入 user:{username} 格式的 thread_id。"
            )

        config = {
            "recursion_limit": self.ctx.recursion_limit,
            "configurable": {"thread_id": thread_id},
        }

        async for event in self._graph.astream_events(
            {"messages": messages},
            context=self._dependencies,
            config=config,
            version="v2",
        ):
            yield event
