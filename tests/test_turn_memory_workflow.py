"""逐轮记忆、上下文预算与 Checkpoint 原文保留测试。"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from react_agent.agent.config import AgentContext
from react_agent.agent.context_management import TurnMemory
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.workflow.graph import build_base_graph


def _messages():
    old = [
        HumanMessage(id="m1", content="比较2025年资本开支，不要编造数字。"),
        AIMessage(
            id="m2",
            content="",
            tool_calls=[{"id": "call-old", "name": "query_internal_knowledge", "args": {}}],
        ),
        ToolMessage(
            id="m3",
            tool_call_id="call-old",
            content=json.dumps(
                {
                    "data": {
                        "results": [
                            {
                                "source_file": "report.pdf",
                                "source_page": 7,
                                "chunk_id": "report::7",
                                "content": "原报告记载2025年资本开支为409亿美元。",
                            }
                        ]
                    }
                },
                ensure_ascii=False,
            ),
        ),
        AIMessage(id="m4", content="背景说明" * 350),
        HumanMessage(id="m5", content="更正：2025年应按420亿美元讨论，不是409亿美元。"),
        AIMessage(id="m6", content="后续讨论" * 350),
    ]
    recent_messages = [
        item
        for index in range(3)
        for item in (
            HumanMessage(id=f"m{7 + 2 * index}", content=f"近期话题{index}"),
            AIMessage(id=f"m{8 + 2 * index}", content=f"近期回答{index}"),
        )
    ]
    current = [
        HumanMessage(id="m13", content="现在请核实最新口径"),
        AIMessage(
            id="m14",
            content="",
            tool_calls=[{"id": "call-now", "name": "query_internal_knowledge", "args": {}}],
        ),
        ToolMessage(id="m15", tool_call_id="call-now", content="当前轮原始证据"),
    ]
    return old, recent_messages, current


def _config() -> AgentContext:
    return AgentContext(
        enable_tools=False,
        max_history_tokens=1000,
        max_input_tokens=20000,
    )


def test_context_defaults_set_history_and_total_budget(monkeypatch) -> None:
    for name in (
        "MAX_HISTORY_TOKENS",
        "MAX_INPUT_TOKENS",
        "MODEL_INPUT_TARGET_TOKENS",
        "HISTORY_RICH_TOKENS",
        "HISTORY_EVIDENCE_TOKENS",
        "HISTORY_RECENT_RICH_TURNS",
    ):
        monkeypatch.delenv(name, raising=False)
    config = AgentContext()
    assert config.max_history_tokens == 64000
    assert config.max_input_tokens == 144000
    assert config.model_input_target_tokens == 132000
    assert config.history_rich_tokens == 44000
    assert config.history_evidence_tokens == 18000
    assert config.history_recent_rich_turns == 4


@pytest.mark.asyncio
async def test_graph_persists_turn_memories_without_replacing_original_messages() -> None:
    old, recent, current = _messages()
    source = old + recent + current

    class FakeModel:
        async def ainvoke(self, messages, *, config=None):
            return AIMessage(
                content="最终回答",
                usage_metadata={"input_tokens": 20, "output_tokens": 4, "total_tokens": 24},
            )

    model = FakeModel()
    dependencies = AgentDependencies(
        config=_config(), model_provider=lambda: model, tools=()
    )
    graph = build_base_graph().compile(checkpointer=MemorySaver())
    assert {
        edge.target for edge in graph.get_graph().edges if edge.source == "prepare_turn"
    } == {"call_model"}
    run_config = {"configurable": {"thread_id": "turn-memory-checkpoint-test"}}
    result = await graph.ainvoke(
        {"messages": source}, context=dependencies, config=run_config
    )
    snapshot = await graph.aget_state(run_config)

    assert [message.content for message in result["messages"][: len(source)]] == [
        message.content for message in source
    ]
    assert result["messages"][-1].content == "最终回答"
    assert snapshot.values["conversation_turn_memories"] == result[
        "conversation_turn_memories"
    ]
    memories = [
        TurnMemory.from_state(raw) for raw in snapshot.values["conversation_turn_memories"]
    ]
    assert [memory.turn_id for memory in memories] == ["m1", "m5", "m7", "m9", "m11"]
    assert [memory.user_messages for memory in memories] == [
        (old[0].content,),
        (old[4].content,),
        ("近期话题0",),
        ("近期话题1",),
        ("近期话题2",),
    ]
    assert memories[0].source_message_ids == ("m1", "m2", "m3", "m4")
    assert memories[1].source_message_ids == ("m5", "m6")
    assert memories[0].final_answer == old[3].content
    assert memories[1].final_answer == old[5].content
    assert [message.content for message in snapshot.values["messages"][: len(source)]] == [
        message.content for message in source
    ]
    assert len(snapshot.values["messages"]) == len(source) + 1
    assert result["llm_call_count"] == 1
