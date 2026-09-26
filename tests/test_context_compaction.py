"""旧轮次摘要的游标、原文保留与预算接入测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from react_agent.agent.config import AgentContext
from react_agent.agent.context_management import ContextBudget, build_model_context
from react_agent.agent.context_management.compaction import (
    accept_compaction,
    compaction_messages,
    evaluate_compaction,
    load_valid_summary,
    plan_compaction,
    preflight_compaction,
    summary_to_state,
)
from react_agent.agent.context_management.budgeting import estimate_message_tokens
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.contracts.state import State
from react_agent.agent.workflow.nodes.lifecycle import compact_history
from react_agent.models import bind_output_token_limit
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


def test_context_defaults_increase_history_and_total_budget(monkeypatch) -> None:
    monkeypatch.delenv("MAX_HISTORY_TOKENS", raising=False)
    monkeypatch.delenv("MAX_INPUT_TOKENS", raising=False)
    monkeypatch.delenv("ENABLE_HISTORY_COMPACTION", raising=False)
    monkeypatch.delenv("HISTORY_COMPACTION_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("HISTORY_COMPACTION_RETRY_NEW_TOKENS", raising=False)
    config = AgentContext()
    assert config.max_history_tokens == 80000
    assert config.max_input_tokens == 96000
    assert config.enable_history_compaction is True
    assert config.history_compaction_max_output_tokens == 768
    assert config.history_compaction_retry_new_tokens == 512


def test_summary_preserves_numbers_negation_sources_and_later_correction() -> None:
    old, recent, current = _messages()
    all_messages = old + recent + current
    state = State(messages=all_messages)
    plan = plan_compaction(state, _config())

    assert plan is not None
    assert list(plan.messages) == old
    assert "m13" not in str(compaction_messages(plan))
    summary = accept_compaction(plan, "用户讨论资本开支，并修订了比较口径。")
    assert summary is not None
    assert summary.source_message_ids == tuple(f"m{number}" for number in range(1, 7))
    assert summary.summarized_through_message_id == "m6"
    assert "不" in "\n".join(summary.exact_excerpts)
    assert "409" in "\n".join(summary.exact_excerpts)
    assert "420" in "\n".join(summary.exact_excerpts)
    assert "report.pdf" in "\n".join(summary.source_references)
    assert "页码 7" in "\n".join(summary.source_references)
    assert "report::7" in "\n".join(summary.source_references)

    state.conversation_summary = summary_to_state(summary)
    state.turn_evidence = [
        {
            "chunk_id": "current::1",
            "source": "current.pdf",
            "page": 1,
            "queries": ["最新口径"],
            "excerpt": "当前证据原文",
            "excerpt_truncated": False,
        }
    ]
    config = SimpleNamespace(enable_history_truncation=True, max_history_tokens=20000)
    context = build_model_context(
        state,
        config,
        system_prompt="系统",
        directive=None,
        budget=ContextBudget(
            max_input_tokens=20000,
            reserved_completion_tokens=100,
            safety_margin_tokens=100,
        ),
    )

    assert context.diagnostics.summary_included is True
    assert context.diagnostics.summarized_message_count == len(old)
    assert context.budget_report is not None
    assert context.budget_report.summary_tokens > 0
    assert context.budget_report.estimated_input_tokens == estimate_message_tokens(
        context.as_list()
    )
    assert "420亿美元" in context.messages[1]["content"]
    assert "report.pdf" in context.messages[1]["content"]
    assert list(context.messages[2:])[:-1] == recent + current[:-1]
    assert "本轮 RAG 证据索引" in context.messages[-1].content
    assert list(state.messages) == all_messages
    assert state.messages[-1].content == "当前轮原始证据"


def test_rejected_overview_does_not_advance_cursor() -> None:
    old, recent, current = _messages()
    plan = plan_compaction(State(messages=old + recent + current), _config())
    assert plan is not None
    assert accept_compaction(plan, "2025年已经确定具体数字。") is None
    assert accept_compaction(plan, "用户不需要核查来源。") is None
    assert accept_compaction(plan, " ") is None
    assert evaluate_compaction(plan, "2025年已经确定具体数字。").reason == "unsafe_overview"
    assert evaluate_compaction(plan, " ").reason == "empty_overview"


def test_critical_excerpts_are_sentence_scoped_and_ignore_benign_bu() -> None:
    old = [
        HumanMessage(id="m1", content="请介绍协作式知识管理。"),
        AIMessage(
            id="m2",
            content=(
                    "知识在团队内不断流动，不仅促进协作，也支持组织学习。"
                    "成员通过对话形成共识。" * 100
                ),
        ),
        HumanMessage(id="m3", content="预算改为420亿元，尚未经过外部核实。"),
        AIMessage(id="m4", content="已记录该更正。"),
    ]
    recent = [
        item
        for index in range(3)
        for item in (
            HumanMessage(id=f"r{index}-h", content="近期问题"),
            AIMessage(id=f"r{index}-a", content="近期回答"),
        )
    ]
    state = State(messages=old + recent + [HumanMessage(id="current", content="当前")])
    plan = plan_compaction(
        state,
        AgentContext(enable_tools=False, max_history_tokens=1, max_input_tokens=20000),
    )

    assert plan is not None
    excerpts = "\n".join(plan.exact_excerpts)
    assert "420亿元" in excerpts
    assert "尚未经过外部核实" in excerpts
    assert "不断流动" not in excerpts
    assert "不仅促进" not in excerpts


def test_compaction_skips_unmatched_completed_tool_exchange() -> None:
    old, recent, current = _messages()
    broken = old[:3] + [
        AIMessage(
            id="m4",
            content="",
            tool_calls=[{"id": "missing", "name": "search", "args": {}}],
        )
    ] + old[4:]
    assert plan_compaction(State(messages=broken + recent + current), _config()) is None


def test_summary_rejects_stale_original_and_preserves_full_history() -> None:
    old, recent, current = _messages()
    plan = plan_compaction(State(messages=old + recent + current), _config())
    assert plan is not None
    summary = accept_compaction(plan, "用户讨论资本开支的比较口径。")
    assert summary is not None
    changed = [old[0].model_copy(update={"content": "用户更新了早期原文"})] + old[1:]
    state = State(
        messages=changed + recent + current,
        conversation_summary=summary_to_state(summary),
    )
    assert load_valid_summary(state.conversation_summary, state.messages) == (None, 0)
    assert plan_compaction(state, _config()) is None

    context = build_model_context(
        state,
        SimpleNamespace(enable_history_truncation=False, max_history_tokens=20000),
        system_prompt="系统",
        directive=None,
    )
    assert context.diagnostics.summary_included is False
    assert context.messages[1] == changed[0]


@pytest.mark.asyncio
async def test_compaction_node_calls_model_once_and_counts_usage() -> None:
    old, recent, current = _messages()

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0
            self.bound_kwargs = {}

        def bind(self, **kwargs):
            self.bound_kwargs = kwargs
            return self

        async def ainvoke(self, messages, *, config=None):
            self.calls += 1
            assert "m13" not in str(messages)
            return AIMessage(
                content="用户讨论资本开支的比较口径。",
                usage_metadata={
                    "input_tokens": 140,
                    "output_tokens": 12,
                    "total_tokens": 152,
                },
            )

    model = FakeModel()
    dependencies = AgentDependencies(
        config=_config(),
        model_provider=lambda: model,
        tools=(),
        model_ref="deepseek/deepseek-flash",
        output_token_limiter=lambda model, *, max_tokens: bind_output_token_limit(
            model,
            model_ref="deepseek/deepseek-flash",
            max_tokens=max_tokens,
        ),
    )
    state = State(messages=old + recent + current)
    update = await compact_history(
        state, SimpleNamespace(context=dependencies), None
    )

    assert model.calls == 1
    assert model.bound_kwargs == {"extra_body": {"max_tokens": 768}}
    assert update["llm_call_count"] == 1
    assert update["prompt_tokens"] == 140
    assert update["completion_tokens"] == 12
    assert update["conversation_summary"]["summarized_through_message_id"] == "m6"
    assert update["compaction_control"]["status"] == "accepted"
    assert update["turn_compaction_usage"]["outcome"] == "accepted"
    assert state.conversation_summary is None
    state.conversation_summary = update["conversation_summary"]
    assert plan_compaction(state, _config()) is None
    assert model.calls == 1


@pytest.mark.asyncio
async def test_compaction_rejects_truncated_model_output_and_counts_usage() -> None:
    old, recent, current = _messages()

    class FakeModel:
        def bind(self, **_kwargs):
            return self

        async def ainvoke(self, _messages, *, config=None):
            return AIMessage(
                content="未完成的摘要",
                response_metadata={"finish_reason": "length"},
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 768,
                    "total_tokens": 868,
                },
            )

    dependencies = AgentDependencies(
        config=_config(),
        model_provider=FakeModel,
        tools=(),
        model_ref="deepseek/deepseek-flash",
        output_token_limiter=lambda model, *, max_tokens: bind_output_token_limit(
            model,
            model_ref="deepseek/deepseek-flash",
            max_tokens=max_tokens,
        ),
    )
    update = await compact_history(
        State(messages=old + recent + current),
        SimpleNamespace(context=dependencies),
        None,
    )

    assert "conversation_summary" not in update
    assert update["compaction_control"]["status"] == "rejected"
    assert update["compaction_control"]["reason"] == "output_truncated"
    assert update["turn_compaction_usage"]["outcome"] == "output_truncated"
    assert update["completion_tokens"] == 768


@pytest.mark.asyncio
async def test_preflight_defers_unprofitable_batch_without_model_call() -> None:
    messages = [
        item
        for index in range(4)
        for item in (
            HumanMessage(id=f"m{index}-h", content="问"),
            AIMessage(id=f"m{index}-a", content="答"),
        )
    ] + [HumanMessage(id="current", content="当前")]
    config = AgentContext(
        enable_tools=False,
        max_history_tokens=1,
        max_input_tokens=20000,
        history_compaction_retry_new_tokens=64,
    )
    state = State(messages=messages)
    plan = plan_compaction(state, config)
    assert plan is not None
    assert preflight_compaction(plan) == "insufficient_projected_savings"

    class FakeModel:
        calls = 0

        async def ainvoke(self, messages, *, config=None):
            self.calls += 1
            raise AssertionError("预判无收益时不应调用模型")

    model = FakeModel()
    dependencies = AgentDependencies(
        config=config,
        model_provider=lambda: model,
        tools=(),
    )
    update = await compact_history(state, SimpleNamespace(context=dependencies), None)

    assert model.calls == 0
    assert update["compaction_control"]["status"] == "deferred"
    assert update["compaction_control"]["reason"] == "insufficient_projected_savings"
    state.compaction_control = update["compaction_control"]
    assert plan_compaction(state, config) is None


@pytest.mark.asyncio
async def test_rejected_compaction_is_cooled_down_until_source_grows() -> None:
    old, recent, current = _messages()

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages, *, config=None):
            self.calls += 1
            return AIMessage(
                content="用户不需要核查来源。",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                },
            )

    model = FakeModel()
    config = _config()
    dependencies = AgentDependencies(
        config=config,
        model_provider=lambda: model,
        tools=(),
    )
    state = State(messages=old + recent + current)
    first = await compact_history(state, SimpleNamespace(context=dependencies), None)

    assert model.calls == 1
    assert first["compaction_control"]["status"] == "rejected"
    assert first["compaction_control"]["reason"] == "unsafe_overview"
    assert first["turn_compaction_usage"]["outcome"] == "unsafe_overview"

    state.compaction_control = first["compaction_control"]
    second = await compact_history(state, SimpleNamespace(context=dependencies), None)
    assert second == {}
    assert model.calls == 1


def test_compaction_advances_only_over_new_complete_turns() -> None:
    old, recent, current = _messages()
    first_plan = plan_compaction(State(messages=old + recent + current), _config())
    assert first_plan is not None
    first = accept_compaction(first_plan, "用户讨论资本开支的比较口径。")
    assert first is not None
    more = [
        HumanMessage(id="m16", content="继续分析"),
        AIMessage(id="m17", content="背景讨论" * 400),
        HumanMessage(id="m18", content="后续话题甲"),
        AIMessage(id="m19", content="后续回答甲"),
        HumanMessage(id="m20", content="后续话题乙"),
        AIMessage(id="m21", content="后续回答乙"),
        HumanMessage(id="m22", content="后续话题丙"),
        AIMessage(id="m23", content="后续回答丙"),
        HumanMessage(id="m24", content="当前新问题"),
    ]
    legacy_summary = summary_to_state(first)
    legacy_summary["exact_excerpts"] = tuple(first.exact_excerpts) + (
        f"[m4] 助手原文：{old[3].content}",
    )
    state = State(
        messages=old + recent + current + more,
        conversation_summary=legacy_summary,
    )
    next_plan = plan_compaction(state, _config())
    assert next_plan is not None
    assert next_plan.previous is not None
    assert next_plan.previous.summarized_through_message_id == first.summarized_through_message_id
    assert next_plan.source_message_ids[: len(first.source_message_ids)] == first.source_message_ids
    assert next_plan.messages[0].id == "m7"
    assert next_plan.messages[-1].id == "m17"
    assert "m24" not in next_plan.source_message_ids
    assert old[3].content not in "\n".join(next_plan.exact_excerpts)
    second = accept_compaction(next_plan, "用户继续围绕资本开支展开讨论。")
    assert second is not None
    assert second.summarized_through_message_id == "m17"
    assert list(state.messages) == old + recent + current + more


def test_budget_can_omit_summary_without_changing_checkpoint() -> None:
    old, recent, _ = _messages()
    current = HumanMessage(id="m13", content="问")
    state = State(messages=old + recent + [current])
    plan = plan_compaction(state, _config())
    assert plan is not None
    summary = accept_compaction(plan, "用户讨论资本开支的比较口径。")
    assert summary is not None
    state.conversation_summary = summary_to_state(summary)
    limit = estimate_message_tokens([{"role": "system", "content": "系统"}, current])
    context = build_model_context(
        state,
        SimpleNamespace(enable_history_truncation=True, max_history_tokens=20000),
        system_prompt="系统",
        directive=None,
        budget=ContextBudget(
            max_input_tokens=limit,
            reserved_completion_tokens=10,
            safety_margin_tokens=5,
        ),
    )
    assert context.as_list() == [{"role": "system", "content": "系统"}, current]
    assert context.diagnostics.summary_included is False
    assert context.budget_report is not None
    assert context.budget_report.summary_omitted_by_budget is True
    assert "summary_budget" in context.budget_report.projection_reasons
    assert state.conversation_summary == summary_to_state(summary)


@pytest.mark.asyncio
async def test_graph_persists_summary_without_replacing_original_messages() -> None:
    old, recent, current = _messages()
    source = old + recent + current

    class FakeModel:
        async def ainvoke(self, messages, *, config=None):
            is_compaction = "context_compaction" in (config or {}).get("tags", ())
            return AIMessage(
                content=(
                    "用户讨论资本开支的比较口径。"
                    if is_compaction
                    else "最终回答"
                ),
                usage_metadata={"input_tokens": 20, "output_tokens": 4, "total_tokens": 24},
            )

    model = FakeModel()
    dependencies = AgentDependencies(
        config=_config(), model_provider=lambda: model, tools=()
    )
    graph = build_base_graph().compile(checkpointer=MemorySaver())
    run_config = {"configurable": {"thread_id": "compaction-checkpoint-test"}}
    result = await graph.ainvoke(
        {"messages": source}, context=dependencies, config=run_config
    )
    snapshot = await graph.aget_state(run_config)

    assert [message.content for message in result["messages"][: len(source)]] == [
        message.content for message in source
    ]
    assert result["messages"][-1].content == "最终回答"
    assert result["conversation_summary"]["summarized_through_message_id"] == "m6"
    persisted_summary, covered_count = load_valid_summary(
        snapshot.values["conversation_summary"], snapshot.values["messages"]
    )
    assert persisted_summary is not None
    assert persisted_summary.summarized_through_message_id == "m6"
    assert covered_count == len(old)
    assert len(snapshot.values["messages"]) == len(source) + 1
    assert result["llm_call_count"] == 2
