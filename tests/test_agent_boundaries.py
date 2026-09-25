from __future__ import annotations

import ast
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph

from react_agent.agent.config import AgentContext
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import InputState, State
from react_agent.agent.service import AgentService
from react_agent.agent.tool_flow.budget import (
    count_attempted_rag_calls_in_current_turn,
    count_successful_rag_calls_in_current_turn,
)
from react_agent.agent.tool_flow.calls import extract_tool_call_ids
from react_agent.agent.tool_flow.payload import bound_tool_payload
from react_agent.agent.workflow.graph import build_base_graph
from react_agent.agent.workflow.nodes import (
    call_model,
    dynamic_tool_node,
    postprocess_tools,
)
from react_agent.agent.workflow.routing import route_after_postprocess
from react_agent.tooling.results import tool_error, tool_success
from react_agent.tooling.retry import with_retry


def test_tool_call_ids_cover_valid_invalid_and_provider_payloads() -> None:
    valid = AIMessage(
        content="",
        tool_calls=[{"id": "valid-1", "name": "search", "args": {}}],
        invalid_tool_calls=[{"id": "invalid-1", "name": "search", "args": "{"}],
    )
    provider_payload = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [{"id": "raw-1", "type": "function", "function": {}}]
        },
    )

    assert extract_tool_call_ids(valid) == ["valid-1", "invalid-1"]
    assert extract_tool_call_ids(provider_payload) == ["raw-1"]


@pytest.mark.asyncio
async def test_last_step_tool_call_returns_a_mergeable_fallback() -> None:
    class LastStepToolCallingModel:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages, config=None):
            return AIMessage(
                id="last-step-tool-call",
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "query_internal_knowledge",
                        "args": {"query": "仍需检索"},
                    }
                ],
                usage_metadata={
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                },
            )

    dependencies = AgentDependencies(
        config=AgentContext(
            enable_history_truncation=False,
        ),
        model_provider=LastStepToolCallingModel,
        tools=(),
    )
    builder = StateGraph(
        State,
        input_schema=InputState,
        context_schema=AgentDependencies,
    )
    builder.add_node("call_model", call_model)
    builder.add_edge("__start__", "call_model")
    builder.add_edge("call_model", "__end__")
    graph = builder.compile()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="测试最后一步兜底")]},
        context=dependencies,
        config={"recursion_limit": 2},
    )

    assert "未继续执行新的工具调用" in result["messages"][-1].content
    assert result["messages"][-1].usage_metadata["total_tokens"] == 7
    assert len(result["debug_log"]) == 1
    assert result["debug_log"][0].startswith("[call_model] is_last_step=True;")
    events = [
        event
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content="测试流式边界兜底")]},
            context=dependencies,
            config={"recursion_limit": 2},
            version="v2",
        )
    ]
    node_ends = [
        event
        for event in events
        if event["event"] == "on_chain_end" and event.get("name") == "call_model"
    ]
    assert len(node_ends) == 1
    assert node_ends[0]["data"]["output"]["termination_reason"] == (
        "UNEXPECTED_RECURSION_BOUNDARY"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("recursion_limit", [4, 5, 6])
async def test_low_graph_step_budget_avoids_tool_call_before_model_invocation(
    recursion_limit: int,
) -> None:
    class Model:
        def __init__(self) -> None:
            self.bound_calls = 0

        def bind_tools(self, _tools):
            self.bound_calls += 1
            return self

        async def ainvoke(self, _messages, config=None):
            if self.bound_calls:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "would-run", "name": "search", "args": {"query": "x"}}
                    ],
                )
            return AIMessage(
                content="根据现有信息直接回答。",
                usage_metadata={
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                },
            )

    model = Model()
    dependencies = AgentDependencies(
        config=AgentContext(enable_history_truncation=False),
        model_provider=lambda: model,
        tools=(),
    )
    result = await build_base_graph().compile().ainvoke(
        {"messages": [HumanMessage(content="测试低图步数")]},
        context=dependencies,
        config={"recursion_limit": recursion_limit},
    )

    assert model.bound_calls == 0
    assert result["messages"][-1].content == "根据现有信息直接回答。"
    assert result["termination_reason"] == "GRAPH_STEP_BUDGET_EXHAUSTED"
    assert result["total_tokens"] == 7
    assert result["llm_call_count"] == 1


@pytest.mark.asyncio
async def test_no_model_is_called_when_no_graph_step_can_finish() -> None:
    dependencies = AgentDependencies(
        config=AgentContext(enable_history_truncation=False),
        model_provider=lambda: pytest.fail("不应调用模型"),
        tools=(),
    )
    state = State(messages=[HumanMessage(content="问题")], remaining_steps=0)
    runtime = SimpleNamespace(context=dependencies)

    with pytest.raises(GraphRecursionError, match="call_model"):
        await call_model(state, runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("current_turn_persisted", [False, True])
async def test_failed_invoke_collects_only_its_checkpointed_model_usage(
    current_turn_persisted: bool,
) -> None:
    class FailingGraph:
        current_input: HumanMessage | None = None

        async def ainvoke(self, input_state, *, context, config):
            self.current_input = input_state["messages"][-1]
            raise GraphRecursionError("test limit")

        async def aget_state(self, config):
            messages = [
                HumanMessage(id="prior", content="问题"),
                AIMessage(content="上轮回答"),
            ]
            if current_turn_persisted:
                messages.extend(
                    [
                        self.current_input,
                        AIMessage(
                            content="",
                            usage_metadata={
                                "input_tokens": 5,
                                "output_tokens": 2,
                                "total_tokens": 7,
                            },
                        ),
                    ]
                )
            return SimpleNamespace(
                values={"messages": messages, "turn_compaction_usage": None}
            )

    graph = FailingGraph()
    dependencies = AgentDependencies(
        config=AgentContext(enable_history_truncation=False),
        model_provider=lambda: None,
        tools=(),
    )
    service = AgentService(dependencies, graph)

    with pytest.raises(GraphRecursionError) as captured:
        await service.invoke([HumanMessage(content="问题")], thread_id="user:test")

    assert graph.current_input.id is not None
    completed = getattr(captured.value, "completed_model_messages", None)
    if current_turn_persisted:
        assert len(completed) == 1
        assert completed[0].usage_metadata["total_tokens"] == 7
    else:
        assert completed is None


@pytest.mark.asyncio
async def test_actual_graph_recursion_error_preserves_completed_model_usage() -> None:
    async def emit_model(state: State) -> dict[str, object]:
        return {
            "messages": [
                AIMessage(
                    content="已产生费用的中间结果",
                    usage_metadata={
                        "input_tokens": 5,
                        "output_tokens": 2,
                        "total_tokens": 7,
                    },
                )
            ]
        }

    async def spin(state: State) -> dict[str, int]:
        return {"step_counter": state.step_counter + 1}

    builder = StateGraph(
        State,
        input_schema=InputState,
        context_schema=AgentDependencies,
    )
    builder.add_node("emit_model", emit_model)
    builder.add_node("spin", spin)
    builder.add_edge("__start__", "emit_model")
    builder.add_edge("emit_model", "spin")
    builder.add_edge("spin", "spin")
    graph = builder.compile(checkpointer=MemorySaver())
    dependencies = AgentDependencies(
        config=AgentContext(
            recursion_limit=11,
            max_model_rounds=1,
            max_tool_batches=1,
            max_tool_retries=1,
            enable_history_truncation=False,
        ),
        model_provider=lambda: None,
        tools=(),
    )
    service = AgentService(dependencies, graph)

    with pytest.raises(GraphRecursionError) as captured:
        await service.invoke([HumanMessage(content="问题")], thread_id="user:loop")

    completed = captured.value.completed_model_messages
    assert len(completed) == 1
    assert completed[0].usage_metadata["total_tokens"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("recursion_limit", [7, 8])
async def test_tool_result_finalizes_while_one_graph_step_is_reserved(
    recursion_limit: int,
) -> None:
    class BoundModel:
        async def ainvoke(self, _messages, config=None):
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "tool-1", "name": "search", "args": {"query": "x"}}
                ],
            )

    class Model:
        def __init__(self) -> None:
            self.bound_calls = 0

        def bind_tools(self, _tools):
            self.bound_calls += 1
            return BoundModel()

        async def ainvoke(self, _messages, config=None):
            return AIMessage(content="根据工具结果完成最终回答。")

    calls: list[str] = []

    def search(query: str) -> str:
        calls.append(query)
        return json.dumps(tool_success(tool_name="search", query=query, data={}))

    model = Model()
    dependencies = AgentDependencies(
        config=AgentContext(enable_history_truncation=False),
        model_provider=lambda: model,
        tools=(StructuredTool.from_function(search, description="测试搜索"),),
    )
    result = await build_base_graph().compile().ainvoke(
        {"messages": [HumanMessage(content="测试工具后的收口")]},
        context=dependencies,
        config={"recursion_limit": recursion_limit},
    )

    assert calls == ["x"]
    assert model.bound_calls == 1
    assert result["turn_tool_batches"] == 1
    assert result["termination_reason"] == "GRAPH_STEP_BUDGET_EXHAUSTED"
    assert result["messages"][-1].content == "根据工具结果完成最终回答。"


@pytest.mark.asyncio
async def test_postprocess_last_step_returns_a_protocol_safe_fallback() -> None:
    pending = AIMessage(
        content="",
        tool_calls=[{"id": "tool-1", "name": "search", "args": {"query": "x"}}],
    )
    tool_result = ToolMessage(
        tool_call_id="tool-1",
        name="search",
        content=json.dumps(tool_success(tool_name="search", query="x", data={})),
    )
    dependencies = AgentDependencies(
        config=AgentContext(enable_history_truncation=False),
        model_provider=lambda: None,
        tools=(),
    )
    builder = StateGraph(
        State,
        input_schema=InputState,
        context_schema=AgentDependencies,
    )
    builder.add_node("postprocess_tools", postprocess_tools)
    builder.add_edge("__start__", "postprocess_tools")
    builder.add_conditional_edges(
        "postprocess_tools", route_after_postprocess, {"__end__": "__end__"}
    )

    result = await builder.compile().ainvoke(
        {"messages": [HumanMessage(content="问题"), pending, tool_result]},
        context=dependencies,
        config={"recursion_limit": 2},
    )

    assert result["termination_reason"] == "UNEXPECTED_RECURSION_BOUNDARY"
    assert isinstance(result["messages"][-1], AIMessage)
    assert "无法继续生成可靠结论" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_tools_do_not_execute_without_room_for_result_processing() -> None:
    calls: list[str] = []

    def search(query: str) -> str:
        calls.append(query)
        return json.dumps(tool_success(tool_name="search", query=query, data={}))

    dependencies = AgentDependencies(
        config=AgentContext(enable_history_truncation=False),
        model_provider=lambda: None,
        tools=(StructuredTool.from_function(search, description="测试搜索"),),
    )
    builder = StateGraph(
        State,
        input_schema=InputState,
        context_schema=AgentDependencies,
    )
    builder.add_node("tools", dynamic_tool_node)
    builder.add_node("postprocess_tools", postprocess_tools)
    builder.add_edge("__start__", "tools")
    builder.add_edge("tools", "postprocess_tools")
    builder.add_conditional_edges(
        "postprocess_tools", route_after_postprocess, {"__end__": "__end__"}
    )
    result = await builder.compile().ainvoke(
        {
            "messages": [
                HumanMessage(content="问题"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "tool-1", "name": "search", "args": {"query": "x"}}
                    ],
                ),
            ]
        },
        context=dependencies,
        config={"recursion_limit": 3},
    )

    assert calls == []
    assert isinstance(result["messages"][-2], ToolMessage)
    assert json.loads(result["messages"][-2].content)["error"]["code"] == (
        "GRAPH_STEP_BUDGET_EXHAUSTED"
    )
    assert "无法继续生成可靠结论" in result["messages"][-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize("recursion_limit", [11, 12, 13, 14])
async def test_business_budget_closes_tool_calls_before_finalizing(
    recursion_limit: int,
) -> None:
    class BoundToolCallingModel:
        async def ainvoke(self, _messages, config=None):
            return AIMessage(
                id="budgeted-tool-call",
                content="",
                tool_calls=[
                    {
                        "id": "budget-call-1",
                        "name": "query_internal_knowledge",
                        "args": {"query": "继续检索"},
                    }
                ],
            )

    class FinalizingModel:
        def __init__(self) -> None:
            self.finalizer_messages = None

        def bind_tools(self, _tools):
            return BoundToolCallingModel()

        async def ainvoke(self, messages, config=None):
            self.finalizer_messages = messages
            return AIMessage(content="根据现有信息生成的最终回答。")

    model = FinalizingModel()
    context = AgentContext(
        recursion_limit=recursion_limit,
        max_model_rounds=1,
        max_tool_batches=1,
        max_tool_retries=1,
        enable_history_truncation=False,
    )
    dependencies = AgentDependencies(
        config=context,
        model_provider=lambda: model,
        tools=(),
    )
    graph = build_base_graph().compile()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="需要多次检索的问题")]},
        context=dependencies,
        config={"recursion_limit": context.recursion_limit},
    )

    messages = result["messages"]
    assert isinstance(messages[-2], ToolMessage)
    closed_payload = json.loads(messages[-2].content)
    assert messages[-2].tool_call_id == "budget-call-1"
    assert closed_payload["error"]["code"] == "TOOL_BUDGET_EXHAUSTED"
    assert messages[-1].content == "根据现有信息生成的最终回答。"
    assert result["termination_reason"] == "MODEL_ROUND_BUDGET_EXHAUSTED"
    assert model.finalizer_messages is not None
    assert any(
        isinstance(message, ToolMessage) and message.tool_call_id == "budget-call-1"
        for message in model.finalizer_messages
    )


def test_recursion_limit_must_cover_business_budget() -> None:
    with pytest.raises(ValueError, match="recursion_limit 不足"):
        AgentContext(recursion_limit=20)


def test_agent_context_environment_loading_is_an_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_MODEL_ROUNDS", "7")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "false")

    context = AgentContext()
    explicit = AgentContext(max_model_rounds=9)

    assert context.max_model_rounds == 7
    assert context.enable_web_search is False
    assert explicit.max_model_rounds == 9


def test_agent_context_contains_only_agent_behavior_configuration() -> None:
    assert {data_field.name for data_field in fields(AgentContext)} == {
        "system_prompt",
        "language",
        "timezone",
        "enable_tools",
        "enable_web_search",
        "max_tool_output_chars",
        "recursion_limit",
        "max_model_rounds",
        "max_tool_batches",
        "max_tool_retries",
        "rag_call_limit",
        "consecutive_failure_threshold",
        "max_history_tokens",
        "max_input_tokens",
        "context_safety_margin_tokens",
        "enable_history_truncation",
        "enable_history_compaction",
        "history_compaction_max_output_tokens",
        "history_compaction_retry_new_tokens",
    }


@pytest.mark.asyncio
async def test_checkpoint_state_survives_graph_recompilation() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.calls: list[list[object]] = []

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages, config=None):
            self.calls.append(list(messages))
            return AIMessage(content=f"answer-{len(self.calls)}")

    model = RecordingModel()
    context = AgentContext(
        enable_history_truncation=False,
    )
    dependencies = AgentDependencies(
        config=context,
        model_provider=lambda: model,
        tools=(),
    )
    checkpointer = MemorySaver()
    run_config = {
        "recursion_limit": context.recursion_limit,
        "configurable": {"thread_id": "checkpoint-module-migration"},
    }

    first_graph = build_base_graph().compile(checkpointer=checkpointer)
    await first_graph.ainvoke(
        {"messages": [HumanMessage(content="first-question")]},
        context=dependencies,
        config=run_config,
    )

    second_graph = build_base_graph().compile(checkpointer=checkpointer)
    result = await second_graph.ainvoke(
        {"messages": [HumanMessage(content="second-question")]},
        context=dependencies,
        config=run_config,
    )

    assert result["messages"][-1].content == "answer-2"
    second_input = model.calls[-1]
    assert any(
        isinstance(message, HumanMessage) and message.content == "first-question"
        for message in second_input
    )
    assert any(
        isinstance(message, AIMessage) and message.content == "answer-1"
        for message in second_input
    )


@pytest.mark.asyncio
async def test_rag_miss_is_finalized_by_tool_free_model() -> None:
    class BoundRagCallingModel:
        async def ainvoke(self, _messages, config=None):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "rag-miss-1",
                        "name": "query_internal_knowledge",
                        "args": {"query": "不存在的资料"},
                    }
                ],
            )

    class FinalizingModel:
        def __init__(self) -> None:
            self.bind_count = 0

        def bind_tools(self, _tools):
            self.bind_count += 1
            return BoundRagCallingModel()

        async def ainvoke(self, _messages, config=None):
            return AIMessage(content="知识库没有提供足够证据，当前无法确认。")

    def query_internal_knowledge(query: str) -> str:
        return json.dumps(
            tool_success(
                tool_name="query_internal_knowledge",
                query=query,
                data={"results": []},
                meta={"has_relevant_content": False},
            ),
            ensure_ascii=False,
        )

    rag_tool = StructuredTool.from_function(
        func=query_internal_knowledge,
        name="query_internal_knowledge",
        description="测试用知识库工具",
    )
    model = FinalizingModel()
    context = AgentContext(
        recursion_limit=20,
        max_model_rounds=3,
        max_tool_batches=2,
        max_tool_retries=1,
        consecutive_failure_threshold=1,
        enable_history_truncation=False,
    )
    dependencies = AgentDependencies(
        config=context,
        model_provider=lambda: model,
        tools=(rag_tool,),
    )

    result = (
        await build_base_graph()
        .compile()
        .ainvoke(
            {"messages": [HumanMessage(content="查询不存在的资料")]},
            context=dependencies,
            config={"recursion_limit": context.recursion_limit},
        )
    )

    assert result["termination_reason"] == "RAG_CONSECUTIVE_MISS"
    assert result["turn_tool_batches"] == 1
    assert result["messages"][-1].content == "知识库没有提供足够证据，当前无法确认。"
    assert model.bind_count == 1


@pytest.mark.asyncio
async def test_failed_tool_batch_uses_bounded_recovery_path() -> None:
    class RecoveringModel:
        def __init__(self) -> None:
            self.agent_calls = 0

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages, config=None):
            self.agent_calls += 1
            if self.agent_calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "failed-tool-1",
                            "name": "failing_tool",
                            "args": {"query": "bad input"},
                        }
                    ],
                )
            return AIMessage(content="工具失败，当前无法完成该操作。")

    def failing_tool(query: str) -> str:
        return json.dumps(
            tool_error(
                tool_name="failing_tool",
                query=query,
                code="BAD_INPUT",
                message="参数无效",
            ),
            ensure_ascii=False,
        )

    model = RecoveringModel()
    context = AgentContext(
        recursion_limit=20,
        max_model_rounds=3,
        max_tool_batches=2,
        max_tool_retries=1,
        enable_history_truncation=False,
    )
    dependencies = AgentDependencies(
        config=context,
        model_provider=lambda: model,
        tools=(
            StructuredTool.from_function(
                func=failing_tool,
                name="failing_tool",
                description="始终返回结构化错误的测试工具",
            ),
        ),
    )

    result = (
        await build_base_graph()
        .compile()
        .ainvoke(
            {"messages": [HumanMessage(content="执行失败工具")]},
            context=dependencies,
            config={"recursion_limit": context.recursion_limit},
        )
    )

    assert result["turn_model_rounds"] == 2
    assert result["turn_tool_batches"] == 1
    assert result["turn_tool_retries"] == 1
    assert result["termination_reason"] is None
    assert result["last_tool_batch_errors"][0]["tool"] == "failing_tool"
    assert result["messages"][-1].content == "工具失败，当前无法完成该操作。"


def test_rag_call_policy_counts_attempts_and_successes_in_current_turn() -> None:
    messages = [
        HumanMessage(content="上一轮"),
        ToolMessage(
            name="query_internal_knowledge",
            tool_call_id="old",
            content=json.dumps(tool_success(tool_name="rag", query="old", data={})),
        ),
        HumanMessage(content="当前轮"),
        ToolMessage(
            name="query_internal_knowledge",
            tool_call_id="failed",
            content=json.dumps(
                tool_error(tool_name="rag", query="new", message="timeout")
            ),
        ),
        ToolMessage(
            name="query_internal_knowledge",
            tool_call_id="success",
            content=json.dumps(tool_success(tool_name="rag", query="new", data={})),
        ),
        ToolMessage(
            name="query_internal_knowledge",
            tool_call_id="blocked",
            content=json.dumps(
                tool_error(
                    tool_name="rag",
                    query="blocked",
                    code="RAG_CALL_BUDGET_EXHAUSTED",
                    message="not executed",
                    meta={"executed": False},
                )
            ),
        ),
    ]

    assert count_attempted_rag_calls_in_current_turn(messages) == 2
    assert count_successful_rag_calls_in_current_turn(messages) == 1


@pytest.mark.asyncio
async def test_rag_call_budget_caps_batch_and_finalizes_with_protocol_closed() -> None:
    class BoundModel:
        def __init__(self, owner) -> None:
            self.owner = owner

        async def ainvoke(self, _messages, config=None):
            self.owner.agent_round += 1
            queries = (
                ["rag-1", "rag-2"]
                if self.owner.agent_round == 1
                else ["rag-3", "rag-4", "rag-5", "rag-6"]
            )
            calls = [
                {
                    "id": f"call-{query}",
                    "name": "query_internal_knowledge",
                    "args": {"query": query},
                }
                for query in queries
            ]
            if self.owner.agent_round == 2:
                calls.append(
                    {
                        "id": "call-search",
                        "name": "search",
                        "args": {"query": "public-data"},
                    }
                )
            return AIMessage(content="", tool_calls=calls)

    class Model:
        def __init__(self) -> None:
            self.agent_round = 0
            self.finalizer_calls = 0

        def bind_tools(self, _tools):
            return BoundModel(self)

        async def ainvoke(self, _messages, config=None):
            self.finalizer_calls += 1
            return AIMessage(content="基于配额内的检索结果生成最终回答。")

    rag_calls: list[str] = []
    search_calls: list[str] = []

    def query_internal_knowledge(query: str) -> str:
        rag_calls.append(query)
        return json.dumps(
            tool_success(
                tool_name="query_internal_knowledge",
                query=query,
                data={
                    "results": [
                        {
                            "content": f"{query} 的可引用正文证据",
                            "source": "report.pdf",
                            "page": 1,
                            "chunk_id": f"chunk-{query}",
                        }
                    ]
                },
                meta={"has_relevant_content": True},
            ),
            ensure_ascii=False,
        )

    def search(query: str) -> str:
        search_calls.append(query)
        return json.dumps(
            tool_success(tool_name="search", query=query, data={}),
            ensure_ascii=False,
        )

    model = Model()
    context = AgentContext(
        recursion_limit=30,
        max_model_rounds=4,
        max_tool_batches=3,
        max_tool_retries=1,
        rag_call_limit=3,
        enable_history_truncation=False,
    )
    dependencies = AgentDependencies(
        config=context,
        model_provider=lambda: model,
        tools=(
            StructuredTool.from_function(
                query_internal_knowledge,
                description="测试 RAG 工具",
            ),
            StructuredTool.from_function(search, description="测试搜索工具"),
        ),
    )

    result = await build_base_graph().compile().ainvoke(
        {"messages": [HumanMessage(content="执行多个检索")]},
        context=dependencies,
        config={"recursion_limit": context.recursion_limit},
    )

    assert sorted(rag_calls) == ["rag-1", "rag-2", "rag-3"]
    assert search_calls == ["public-data"]
    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert {message.tool_call_id for message in tool_messages} == {
        "call-rag-1",
        "call-rag-2",
        "call-rag-3",
        "call-rag-4",
        "call-rag-5",
        "call-rag-6",
        "call-search",
    }
    rag_messages = [
        message
        for message in tool_messages
        if message.name == "query_internal_knowledge"
    ]
    assert {message.tool_call_id for message in rag_messages} == {
        "call-rag-1",
        "call-rag-2",
        "call-rag-3",
        "call-rag-4",
        "call-rag-5",
        "call-rag-6",
    }
    budget_errors = [
        json.loads(message.content)
        for message in rag_messages
        if json.loads(message.content).get("error")
        and json.loads(message.content)["error"]["code"]
        == "RAG_CALL_BUDGET_EXHAUSTED"
    ]
    assert len(budget_errors) == 3
    assert all(item["meta"]["executed"] is False for item in budget_errors)
    assert result["termination_reason"] == "RAG_CALL_BUDGET_EXHAUSTED"
    assert model.agent_round == 2
    assert model.finalizer_calls == 1
    assert result["messages"][-1].content == "基于配额内的检索结果生成最终回答。"
    assert sum(run["executed"] is True for run in result["tool_runs"]) == 4
    assert sum(run["executed"] is False for run in result["tool_runs"]) == 3


def test_tool_payload_bounding_preserves_envelope_and_traceability() -> None:
    payload = tool_success(
        tool_name="query_internal_knowledge",
        query="query",
        data={
            "results": [
                {
                    "content": "a" * 800,
                    "source": "one.pdf",
                    "page": 1,
                    "chunk_id": "one::1",
                },
                {
                    "content": "b" * 800,
                    "source": "two.pdf",
                    "page": 2,
                    "chunk_id": "two::2",
                },
            ]
        },
        meta={"has_relevant_content": True},
    )

    encoded = bound_tool_payload(json.dumps(payload), max_chars=700)
    bounded = json.loads(encoded)

    assert len(encoded) <= 700
    assert bounded["ok"] is True
    assert bounded["tool"] == "query_internal_knowledge"
    assert bounded["meta"]["truncated"] is True
    assert bounded["meta"]["total_results"] == 2
    assert bounded["meta"]["retrieval_hit"] is True
    assert bounded["meta"]["has_relevant_content"] is True
    assert bounded["data"]["has_relevant_content"] is True
    assert len(bounded["data"]["results"]) == 2
    assert [item["chunk_id"] for item in bounded["data"]["results"]] == [
        "one::1",
        "two::2",
    ]
    assert all(item["content"] for item in bounded["data"]["results"])


def test_rag_payload_reports_budget_error_when_no_evidence_can_fit() -> None:
    payload = tool_success(
        tool_name="query_internal_knowledge",
        query="query",
        data={
            "results": [
                {
                    "source": "very-long-file-name.pdf",
                    "page": 1,
                    "chunk_id": "very-long-id",
                    "content": "important evidence",
                }
            ],
            "has_relevant_content": True,
        },
        meta={"has_relevant_content": True},
    )

    encoded = bound_tool_payload(json.dumps(payload), max_chars=100)
    bounded = json.loads(encoded)

    assert len(encoded) <= 100
    assert bounded["ok"] is False
    assert bounded["error"]


def test_rag_payload_removes_duplicate_source_prefix_before_bounding() -> None:
    payload = tool_success(
        tool_name="query_internal_knowledge",
        query="query",
        data={
            "results": [
                {
                    "source": "report.pdf",
                    "page": 3,
                    "chunk_id": "report::3",
                    "content": "[来源：report.pdf  第 3 页]\n关键数据 520 亿美元",
                }
            ]
        },
        meta={"has_relevant_content": True},
    )

    bounded = json.loads(bound_tool_payload(json.dumps(payload), max_chars=500))

    assert bounded["data"]["results"][0]["content"] == "关键数据 520 亿美元"


def test_rag_payload_keeps_three_long_table_chunks_under_default_budget() -> None:
    payload = tool_success(
        tool_name="query_internal_knowledge",
        query="2026 年资本开支",
        data={
            "results": [
                {
                    "source": f"report-{index}.pdf",
                    "page": index + 1,
                    "chunk_id": f"report-{index}::table",
                    "content": '| 项目 | 数值 |\n| 资本开支 | "520–560 亿美元" |\n'
                    * 300,
                }
                for index in range(3)
            ]
        },
        meta={"has_relevant_content": True},
    )

    encoded = bound_tool_payload(json.dumps(payload, ensure_ascii=False), 4000)
    bounded = json.loads(encoded)

    assert len(encoded) <= 4000
    assert bounded["meta"]["visible_results"] == 3
    assert [item["chunk_id"] for item in bounded["data"]["results"]] == [
        f"report-{index}::table" for index in range(3)
    ]
    assert all(item["content_truncated"] for item in bounded["data"]["results"])


@pytest.mark.asyncio
async def test_tool_retry_uses_public_result_contract() -> None:
    attempts = 0

    @with_retry(tool_name="test", max_retries=1, base_delay=0)
    async def flaky(query: str) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary")
        return tool_success(tool_name="test", query=query, data={"done": True})

    payload = json.loads(await flaky("hello"))

    assert attempts == 2
    assert payload["ok"] is True
    assert payload["meta"]["attempt"] == 1


def _python_sources(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def _imports_from(source_path: Path, module_prefix: str) -> list[str]:
    imports: list[str] = []
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            module_prefix
        ):
            imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith(module_prefix)
            )
    return imports


def test_utils_package_is_fully_removed() -> None:
    package_root = Path(__file__).parent.parent / "react_agent"

    assert not (package_root / "utils").exists()
    for source_path in _python_sources(package_root):
        assert "react_agent.utils" not in source_path.read_text(encoding="utf-8")


def test_context_legacy_entrypoints_are_fully_removed() -> None:
    project_root = Path(__file__).parent.parent
    agent_root = project_root / "react_agent" / "agent"
    removed_paths = (
        agent_root / "modeling" / "history.py",
        agent_root / "tool_flow" / "evidence.py",
    )
    removed_modules = (
        "react_agent.agent.modeling.history",
        "react_agent.agent.tool_flow.evidence",
    )

    assert all(not path.exists() for path in removed_paths)
    violations: dict[str, list[str]] = {}
    for source_path in _python_sources(project_root):
        imports = [
            imported
            for module in removed_modules
            for imported in _imports_from(source_path, module)
            if imported == module or imported.startswith(f"{module}.")
        ]
        if imports:
            violations[str(source_path.relative_to(project_root))] = imports
    assert violations == {}


def test_internal_symbols_are_not_imported_across_project_modules() -> None:
    project_root = Path(__file__).parent.parent
    source_roots = (
        project_root / "react_agent",
        project_root / "api",
        project_root / "knowledge",
        project_root / "eval",
        project_root / "scripts",
    )
    violations: list[str] = []
    for source_root in source_roots:
        for source_path in _python_sources(source_root):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not (node.module or "").startswith("react_agent."):
                    continue
                for imported in node.names:
                    if imported.name.startswith("_"):
                        violations.append(
                            f"{source_path.relative_to(project_root)} imports "
                            f"{node.module}.{imported.name}"
                        )
    assert violations == []


def test_agent_does_not_depend_on_outbound_adapter_implementations() -> None:
    package_root = Path(__file__).parent.parent / "react_agent"
    forbidden_prefixes = (
        "react_agent.tools",
        "react_agent.models",
        "react_agent.infrastructure",
    )
    violations: dict[str, list[str]] = {}
    for path in _python_sources(package_root / "agent"):
        imports = [
            imported
            for prefix in forbidden_prefixes
            for imported in _imports_from(path, prefix)
        ]
        if imports:
            violations[str(path.relative_to(package_root))] = imports
    assert violations == {}


def test_agent_nodes_remain_thin_orchestration_adapters() -> None:
    project_root = Path(__file__).parent.parent
    nodes_root = project_root / "react_agent" / "agent" / "workflow" / "nodes"
    forbidden_dependencies = (
        "langgraph.prebuilt",
        "react_agent.tooling",
        "tiktoken",
    )

    imports = [
        imported
        for nodes_path in _python_sources(nodes_root)
        for prefix in forbidden_dependencies
        for imported in _imports_from(nodes_path, prefix)
    ]

    assert imports == []


def test_agent_root_contains_expected_flat_modules() -> None:
    project_root = Path(__file__).parent.parent
    agent_root = project_root / "react_agent" / "agent"
    allowed_root_modules = {
        "__init__.py",
        "config.py",
        "model_execution.py",
        "policies.py",
        "prompts.py",
        "service.py",
        "time.py",
    }

    assert {path.name for path in agent_root.glob("*.py")} == allowed_root_modules


def test_removed_agent_module_paths_are_not_imported() -> None:
    project_root = Path(__file__).parent.parent
    removed_modules = {
        "react_agent.agent.context",
        "react_agent.agent.dependencies",
        "react_agent.agent.graph",
        "react_agent.agent.application",
        "react_agent.agent.configuration",
        "react_agent.agent.modeling",
        "react_agent.agent.nodes",
        "react_agent.agent.policies.execution",
        "react_agent.agent.prompting",
        "react_agent.agent.state",
        "react_agent.agent.support",
        "react_agent.agent.tool_calls",
        "react_agent.agent.tool_policy",
        "react_agent.agent.usage",
    }
    violations: dict[str, list[str]] = {}

    for source_path in _python_sources(project_root):
        imports = [
            imported
            for removed in removed_modules
            for imported in _imports_from(source_path, removed)
            if imported == removed or imported.startswith(f"{removed}.")
        ]
        if imports:
            violations[str(source_path.relative_to(project_root))] = imports

    assert violations == {}


def test_agent_subpackages_follow_dependency_direction() -> None:
    project_root = Path(__file__).parent.parent
    agent_root = project_root / "react_agent" / "agent"
    constraints = {
        agent_root / "service.py": (
            "react_agent.agent.model_execution",
            "react_agent.agent.tool_flow",
            "react_agent.agent.workflow",
        ),
        agent_root / "contracts": (
            "react_agent.agent.model_execution",
            "react_agent.agent.tool_flow",
            "react_agent.agent.workflow",
        ),
        agent_root / "config.py": (
            "react_agent.agent.contracts",
            "react_agent.agent.model_execution",
            "react_agent.agent.tool_flow",
            "react_agent.agent.workflow",
        ),
        agent_root / "policies.py": (
            "react_agent.agent.config",
            "react_agent.agent.contracts",
            "react_agent.agent.model_execution",
            "react_agent.agent.tool_flow",
            "react_agent.agent.workflow",
        ),
        agent_root / "prompts.py": (
            "react_agent.agent.config",
            "react_agent.agent.contracts",
            "react_agent.agent.model_execution",
            "react_agent.agent.policies",
            "react_agent.agent.tool_flow",
            "react_agent.agent.workflow",
        ),
        agent_root / "model_execution.py": ("react_agent.agent.workflow",),
        agent_root / "tool_flow": (
            "react_agent.agent.model_execution",
            "react_agent.agent.workflow",
        ),
    }
    violations: dict[str, list[str]] = {}

    for source_root, forbidden_prefixes in constraints.items():
        source_paths = [source_root] if source_root.is_file() else _python_sources(source_root)
        for source_path in source_paths:
            imports = [
                imported
                for prefix in forbidden_prefixes
                for imported in _imports_from(source_path, prefix)
            ]
            if imports:
                violations[str(source_path.relative_to(project_root))] = imports

    assert violations == {}


def test_infrastructure_does_not_depend_on_agent() -> None:
    package_root = Path(__file__).parent.parent / "react_agent"
    infrastructure_roots = (
        package_root / "infrastructure",
        package_root / "models",
        package_root / "observability",
        package_root / "rag" / "infrastructure",
        package_root / "conversations" / "infrastructure",
    )
    violations: dict[str, list[str]] = {}
    for root in infrastructure_roots:
        for path in _python_sources(root):
            imports = _imports_from(path, "react_agent.agent")
            if imports:
                violations[str(path.relative_to(package_root))] = imports
    assert violations == {}
