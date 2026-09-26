from types import SimpleNamespace

import pytest

from react_agent.observability.progress import (
    AgentProgress,
    consume_progress_events,
    progress_from_event,
    root_graph_output,
)


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        ("call_model", "正在分析问题并规划下一步……"),
        ("finalize_model", "正在整理最终回答……"),
        ("compact_history", "正在整理较早的对话上下文……"),
    ],
)
def test_model_start_progress_uses_graph_node(node: str, expected: str) -> None:
    progress = progress_from_event(
        {
            "event": "on_chat_model_start",
            "metadata": {"langgraph_node": node},
        }
    )

    assert progress is not None
    assert progress.label == expected


def test_model_stream_progress_requires_visible_non_compaction_content() -> None:
    assert progress_from_event(
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": SimpleNamespace(content="结论")},
        }
    ) == AgentProgress(stage="model_stream", label="正在生成回答……")
    assert (
        progress_from_event(
            {
                "event": "on_chat_model_stream",
                "tags": ["context_compaction"],
                "data": {"chunk": SimpleNamespace(content="历史摘要")},
            }
        )
        is None
    )
    assert (
        progress_from_event(
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="")},
            }
        )
        is None
    )


@pytest.mark.parametrize(
    ("event_name", "expected"),
    [
        ("on_tool_start", "正在查询内部知识库……"),
        ("on_tool_end", "知识库查询完成，正在分析结果……"),
    ],
)
def test_known_tool_progress_does_not_expose_payload(
    event_name: str,
    expected: str,
) -> None:
    progress = progress_from_event(
        {
            "event": event_name,
            "name": "query_internal_knowledge",
            "data": {"input": {"query": "private query"}},
        }
    )

    assert progress is not None
    assert progress.label == expected
    assert "private query" not in progress.label


def test_unknown_tool_name_is_bounded_and_sanitized() -> None:
    progress = progress_from_event(
        {
            "event": "on_tool_start",
            "name": "tool<script>alert(1)</script>" * 10,
        }
    )

    assert progress is not None
    assert "<" not in progress.label
    assert len(progress.stage) <= 59


def test_root_chain_end_exposes_progress_and_final_state() -> None:
    event = {
        "event": "on_chain_end",
        "name": "LangGraph",
        "parent_ids": [],
        "data": {"output": {"messages": ["answer"], "total_tokens": 12}},
    }

    assert progress_from_event(event) == AgentProgress(
        stage="complete",
        label="Agent 已完成回答",
        state="complete",
    )
    assert root_graph_output(event) == {
        "messages": ["answer"],
        "total_tokens": 12,
    }


def test_nested_chain_end_is_not_treated_as_graph_completion() -> None:
    event = {
        "event": "on_chain_end",
        "parent_ids": ["root"],
        "data": {"output": {"messages": ["partial"]}},
    }

    assert progress_from_event(event) is None
    assert root_graph_output(event) is None


@pytest.mark.asyncio
async def test_consume_progress_events_deduplicates_and_returns_final_state() -> None:
    async def events():
        yield {
            "event": "on_chat_model_start",
            "metadata": {"langgraph_node": "call_model"},
        }
        yield {
            "event": "on_chat_model_start",
            "metadata": {"langgraph_node": "call_model"},
        }
        yield {
            "event": "on_chain_end",
            "parent_ids": [],
            "data": {"output": {"messages": ["answer"], "total_tokens": 12}},
        }

    emitted: list[AgentProgress] = []
    result = await consume_progress_events(events(), emitted.append)

    assert [progress.stage for progress in emitted] == ["call_model", "complete"]
    assert result == {"messages": ["answer"], "total_tokens": 12}


@pytest.mark.asyncio
async def test_consume_progress_events_requires_root_final_state() -> None:
    async def events():
        yield {
            "event": "on_chain_end",
            "parent_ids": ["root"],
            "data": {"output": {"messages": ["partial"]}},
        }

    with pytest.raises(RuntimeError, match="未返回最终图状态"):
        await consume_progress_events(events(), lambda _progress: None)
