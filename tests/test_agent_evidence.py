from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from react_agent.agent.config import AgentContext
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import State
from react_agent.agent.context_management.evidence import merge_visible_evidence
from react_agent.agent.context_management.evidence import (
    merge_historical_evidence,
    render_historical_evidence_index,
)
from react_agent.agent.context_management.builder import build_model_context
from react_agent.agent.context_management.contracts import ContextBudget
from react_agent.agent.tool_flow import bound_tool_payload, parse_tool_batch
from react_agent.agent.workflow.nodes.lifecycle import prepare_turn
from react_agent.agent.workflow.nodes.model import finalize_model
from react_agent.agent.workflow.nodes.tools import postprocess_tools
from knowledge.contracts import RetrievedChunk, RetrievalResult
from react_agent.tools.rag import create_rag_tool
from react_agent.agent.workflow.graph import build_base_graph
from react_agent.tooling.results import tool_success


def _rag_message(query: str, call_id: str, items: list[dict]) -> ToolMessage:
    payload = tool_success(
        tool_name="query_internal_knowledge",
        query=query,
        data={"results": items, "has_relevant_content": bool(items)},
        meta={"has_relevant_content": bool(items)},
    )
    return ToolMessage(
        name="query_internal_knowledge",
        tool_call_id=call_id,
        content=bound_tool_payload(json.dumps(payload, ensure_ascii=False), 600),
    )


def test_evidence_merges_visible_chunks_across_batches_without_mutation() -> None:
    first = _rag_message(
        "资本开支",
        "call-1",
        [{"source": "a.pdf", "page": 1, "chunk_id": "a::1", "content": "520 亿"}],
    )
    second = _rag_message(
        "资本开支指引",
        "call-2",
        [
            {"source": "a.pdf", "page": 1, "chunk_id": "a::1", "content": "520 亿"},
            {"source": "b.pdf", "page": 2, "chunk_id": "b::2", "content": "560 亿"},
        ],
    )

    earlier, omitted = merge_visible_evidence([], [first])
    merged, later_omitted = merge_visible_evidence(earlier, [second])

    assert omitted == later_omitted == 0
    assert [record["chunk_id"] for record in merged] == ["a::1", "b::2"]
    assert merged[0]["queries"] == ["资本开支", "资本开支指引"]
    assert earlier[0]["queries"] == ["资本开支"]
    assert merged[1]["source"] == "b.pdf"
    assert merged[1]["page"] == 2


def test_evidence_ledger_reports_capacity_without_claiming_absence() -> None:
    message = _rag_message(
        "query",
        "call-1",
        [
            {"source": "a.pdf", "page": 1, "chunk_id": "a", "content": "事实 A"},
            {"source": "b.pdf", "page": 2, "chunk_id": "b", "content": "事实 B"},
        ],
    )

    records, omitted = merge_visible_evidence([], [message], max_records=1)

    assert len(records) == 1
    assert omitted == 1


def test_evidence_ledger_counts_results_hidden_by_payload_limit() -> None:
    message = _rag_message(
        "query",
        "call-1",
        [
            {
                "source": f"report-{index}.pdf",
                "page": index + 1,
                "chunk_id": f"chunk-{index}",
                "content": "事实" * 100,
            }
            for index in range(3)
        ],
    )
    bounded = json.loads(message.content)

    records, omitted = merge_visible_evidence([], [message])

    assert omitted == bounded["meta"]["total_results"] - len(records)


def test_historical_evidence_keeps_sources_without_copying_body() -> None:
    message = _rag_message(
        "全球锂矿储量",
        "call-1",
        [
            {
                "source": "lithium.pdf",
                "page": 2,
                "chunk_id": "lithium::2",
                "content": "全球储量约 2800 万金属吨",
            }
        ],
    )

    records, omitted = merge_historical_evidence([], [message])
    rendered = render_historical_evidence_index(records)

    assert omitted == 0
    assert records[0]["chunk_id"] == "lithium::2"
    assert records[0]["source"] == "lithium.pdf"
    assert records[0]["page"] == 2
    assert "excerpt" not in records[0]
    assert "2800" not in rendered
    assert "lithium.pdf" in rendered
    assert "lithium::2" in rendered


@pytest.mark.asyncio
async def test_postprocess_writes_evidence_and_resets_it_next_turn() -> None:
    message = _rag_message(
        "query",
        "call-1",
        [{"source": "a.pdf", "page": 1, "chunk_id": "a", "content": "事实 A"}],
    )
    dependencies = AgentDependencies(
        config=AgentContext(), model_provider=lambda: object(), tools=()
    )
    state = State(messages=[HumanMessage(content="问题"), message])

    update = await postprocess_tools(state, SimpleNamespace(context=dependencies))

    assert update["turn_evidence"][0]["chunk_id"] == "a"
    assert update["conversation_evidence"][0]["chunk_id"] == "a"
    assert update["tool_runs"][0]["sources"] == [
        {"source_file": "a.pdf", "source_page": 1, "chunk_id": "a"}
    ]
    assert "data" not in update["last_tool_result"]
    reset = await prepare_turn(
        State(
            turn_evidence=update["turn_evidence"],
            conversation_evidence=update["conversation_evidence"],
        ),
        SimpleNamespace(context=dependencies),
    )
    assert reset["turn_evidence"] == []
    assert reset["turn_evidence_omitted_count"] == 0
    assert "conversation_evidence" not in reset


@pytest.mark.asyncio
async def test_prepare_turn_backfills_sources_from_legacy_checkpoint() -> None:
    message = _rag_message(
        "历史查询",
        "legacy-call",
        [
            {
                "source": "legacy.pdf",
                "page": 4,
                "chunk_id": "legacy::4",
                "content": "历史证据",
            }
        ],
    )
    state = State(
        messages=[HumanMessage(content="旧问题"), message, AIMessage(content="旧回答")]
    )

    dependencies = AgentDependencies(
        config=AgentContext(), model_provider=lambda: object(), tools=()
    )
    update = await prepare_turn(state, SimpleNamespace(context=dependencies))

    assert update["conversation_evidence"][0]["chunk_id"] == "legacy::4"
    assert update["conversation_evidence_scanned_message_count"] == 3


def test_historical_sources_remain_visible_after_rag_turn_is_trimmed() -> None:
    tool_message = _rag_message(
        "全球锂矿储量",
        "rag-call",
        [
            {
                "source": "lithium.pdf",
                "page": 2,
                "chunk_id": "lithium::2",
                "content": "全球储量约 2800 万金属吨",
            }
        ],
    )
    records, _ = merge_historical_evidence([], [tool_message])
    old_turn = [
        HumanMessage(content="查询全球锂矿储量"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "rag-call",
                    "name": "query_internal_knowledge",
                    "args": {"query": "全球锂矿储量"},
                }
            ],
        ),
        tool_message,
        AIMessage(content="根据知识库回答。"),
    ]
    recent: list = []
    for index in range(3):
        recent.extend(
            [
                HumanMessage(content=f"普通问题 {index}"),
                AIMessage(content="普通回答" * 80),
            ]
        )
    current = HumanMessage(content="回顾此前检索来源")
    state = State(
        messages=old_turn + recent + [current],
        conversation_evidence=records,
    )

    context = build_model_context(
        state,
        AgentContext(max_history_tokens=300),
        system_prompt="系统提示",
        directive=None,
        budget=ContextBudget(
            max_input_tokens=10_000,
            reserved_completion_tokens=1_000,
            safety_margin_tokens=0,
        ),
    )

    assert tool_message not in context.messages
    historical = next(
        item
        for item in context.messages
        if isinstance(item, dict)
        and "历史 RAG 来源索引" in str(item.get("content"))
    )
    assert "lithium.pdf" in historical["content"]
    assert "lithium::2" in historical["content"]
    assert context.budget_report is not None
    assert context.budget_report.historical_evidence_tokens > 0


@pytest.mark.asyncio
async def test_finalizer_receives_evidence_as_transient_tool_data() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.messages = None

        async def ainvoke(self, messages, config=None):
            self.messages = messages
            return AIMessage(content="答案")

    message = _rag_message(
        "query",
        "call-1",
        [{"source": "a.pdf", "page": 1, "chunk_id": "a", "content": "事实 A"}],
    )
    records, _ = merge_visible_evidence([], [message])
    state = State(
        messages=[
            HumanMessage(content="问题"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "query_internal_knowledge",
                        "args": {},
                    }
                ],
            ),
            message,
        ],
        turn_evidence=records,
        termination_reason="RAG_CALL_BUDGET_EXHAUSTED",
    )
    model = RecordingModel()
    dependencies = AgentDependencies(
        config=AgentContext(enable_history_truncation=False),
        model_provider=lambda: model,
        tools=(),
    )

    await finalize_model(state, SimpleNamespace(context=dependencies))

    assert model.messages is not None
    tool_input = next(item for item in model.messages if isinstance(item, ToolMessage))
    assert "本轮 RAG 证据索引" in tool_input.content
    assert '"chunk_id":"a"' in tool_input.content
    assert "本轮 RAG 证据索引" not in message.content


@pytest.mark.asyncio
async def test_graph_merges_two_rag_batches_before_finalization() -> None:
    class ToolCallingModel:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages, config=None):
            self.calls += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call-{self.calls}",
                        "name": "query_internal_knowledge",
                        "args": {"query": f"query-{self.calls}"},
                    }
                ],
            )

    class FinalizingModel:
        def __init__(self) -> None:
            self.tool_model = ToolCallingModel()
            self.final_input = None

        def bind_tools(self, tools):
            return self.tool_model

        async def ainvoke(self, messages, config=None):
            self.final_input = messages
            return AIMessage(content="已汇总。")

    def query_internal_knowledge(query: str) -> str:
        number = query.rsplit("-", 1)[-1]
        return json.dumps(
            tool_success(
                tool_name="query_internal_knowledge",
                query=query,
                data={
                    "results": [
                        {
                            "source": f"report-{number}.pdf",
                            "page": int(number),
                            "chunk_id": f"chunk-{number}",
                            "content": f"第 {number} 份报告的证据",
                        }
                    ]
                },
                meta={"has_relevant_content": True},
            ),
            ensure_ascii=False,
        )

    model = FinalizingModel()
    context = AgentContext(
        max_model_rounds=3,
        max_tool_batches=2,
        max_tool_retries=1,
        recursion_limit=20,
        enable_history_truncation=False,
    )
    tool = StructuredTool.from_function(
        func=query_internal_knowledge,
        name="query_internal_knowledge",
        description="测试 RAG",
    )
    dependencies = AgentDependencies(
        config=context, model_provider=lambda: model, tools=(tool,)
    )

    result = (
        await build_base_graph()
        .compile()
        .ainvoke(
            {"messages": [HumanMessage(content="跨文档汇总")]},
            context=dependencies,
            config={"recursion_limit": context.recursion_limit},
        )
    )

    assert [record["chunk_id"] for record in result["turn_evidence"]] == [
        "chunk-1",
        "chunk-2",
    ]
    assert result["messages"][-1].content == "已汇总。"
    assert model.final_input is not None
    assert any(
        isinstance(item, ToolMessage)
        and '"chunk_id":"chunk-1"' in item.content
        and '"chunk_id":"chunk-2"' in item.content
        for item in model.final_input
    )


@pytest.mark.asyncio
async def test_evidence_does_not_leak_into_next_checkpointed_turn() -> None:
    class ToolCallingModel:
        async def ainvoke(self, messages, config=None):
            latest_question = next(
                message.content
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            )
            if latest_question == "second":
                return AIMessage(content="第二轮答案")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "query_internal_knowledge",
                        "args": {"query": "first"},
                    }
                ],
            )

    class FinalizingModel:
        def __init__(self) -> None:
            self.tool_model = ToolCallingModel()

        def bind_tools(self, tools):
            return self.tool_model

        async def ainvoke(self, messages, config=None):
            return AIMessage(content="第一轮答案")

    def query_internal_knowledge(query: str) -> str:
        return json.dumps(
            tool_success(
                tool_name="query_internal_knowledge",
                query=query,
                data={
                    "results": [
                        {
                            "source": "first.pdf",
                            "page": 1,
                            "chunk_id": "first::1",
                            "content": "第一轮证据",
                        }
                    ]
                },
                meta={"has_relevant_content": True},
            ),
            ensure_ascii=False,
        )

    context = AgentContext(
        max_model_rounds=2,
        max_tool_batches=1,
        max_tool_retries=1,
        recursion_limit=20,
        enable_history_truncation=False,
    )
    tool = StructuredTool.from_function(
        func=query_internal_knowledge,
        name="query_internal_knowledge",
        description="测试 RAG",
    )
    dependencies = AgentDependencies(
        config=context, model_provider=FinalizingModel, tools=(tool,)
    )
    graph = build_base_graph().compile(checkpointer=MemorySaver())
    run_config = {
        "recursion_limit": context.recursion_limit,
        "configurable": {"thread_id": "evidence-turn-isolation"},
    }

    first = await graph.ainvoke(
        {"messages": [HumanMessage(content="first")]},
        context=dependencies,
        config=run_config,
    )
    second = await graph.ainvoke(
        {"messages": [HumanMessage(content="second")]},
        context=dependencies,
        config=run_config,
    )

    assert first["turn_evidence"][0]["chunk_id"] == "first::1"
    assert second["turn_evidence"] == []
    assert second["conversation_evidence"][0]["chunk_id"] == "first::1"
    assert second["messages"][-1].content == "第二轮答案"


@pytest.mark.asyncio
async def test_payload_overflow_is_terminal_error_not_rag_miss() -> None:
    payload = tool_success(
        tool_name="query_internal_knowledge",
        query="query",
        data={
            "results": [
                {
                    "source": "long-file-name.pdf",
                    "page": 1,
                    "chunk_id": "long-id",
                    "content": "事实",
                }
            ]
        },
        meta={"has_relevant_content": True},
    )
    bounded = bound_tool_payload(json.dumps(payload), 100)
    message = ToolMessage(
        name="query_internal_knowledge", tool_call_id="call-1", content=bounded
    )

    batch = parse_tool_batch(
        [HumanMessage(content="问题"), message], timezone="Asia/Shanghai"
    )

    assert batch is not None
    assert batch.error_count == 1
    assert batch.consecutive_rag_misses == 0
    dependencies = AgentDependencies(
        config=AgentContext(), model_provider=lambda: object(), tools=()
    )
    update = await postprocess_tools(
        State(messages=[HumanMessage(content="问题"), message]),
        SimpleNamespace(context=dependencies),
    )
    assert update["termination_reason"] == "EVIDENCE_OUTPUT_BUDGET_EXHAUSTED"


@pytest.mark.asyncio
async def test_real_rag_tool_result_is_json_visible_to_ledger() -> None:
    class FakeRetrievalService:
        async def search(self, query: str, top_k: int):
            assert top_k == 3
            return RetrievalResult(
                query=query,
                chunks=(
                    RetrievedChunk(
                        content="预计资本开支为 520 亿美元",
                        source_file="report.pdf",
                        source_page=2,
                        chunk_id="report::2",
                    ),
                ),
            )

    tool = create_rag_tool(
        retrieval_service_provider=FakeRetrievalService,
        max_retries=0,
        timeout=5,
    )
    result = await tool.ainvoke(
        {
            "type": "tool_call",
            "id": "call-1",
            "name": "query_internal_knowledge",
            "args": {"query": "资本开支"},
        }
    )
    assert isinstance(result, ToolMessage)
    bounded = ToolMessage(
        name="query_internal_knowledge",
        tool_call_id="call-1",
        content=bound_tool_payload(result.content, 600),
    )

    records, omitted = merge_visible_evidence([], [bounded])

    assert omitted == 0
    assert records[0]["chunk_id"] == "report::2"
    assert records[0]["page"] == 2
    assert records[0]["excerpt"] == "预计资本开支为 520 亿美元"
