"""Agent 上下文管理模块的离线边界测试。"""

from __future__ import annotations

import json
import builtins
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from react_agent.agent.context_management import (
    ContextBudget,
    ContextBudgetExceeded,
    ConversationSummary,
    build_model_context,
    latest_turn_ai_messages,
)
from react_agent.agent.context_management.segmentation import segment_current_turn
from react_agent.agent.context_management.budgeting import estimate_message_tokens
from react_agent.agent.context_management.evidence import (
    attach_evidence_index,
    render_evidence_index,
)
from react_agent.agent.context_management.projection import (
    project_completed_tool_messages,
    project_current_tool_messages,
    project_tool_content,
)
from react_agent.agent.context_management.protocol import (
    ToolProtocolError,
    validate_tool_exchanges,
)
from react_agent.agent.contracts.state import State
from react_agent.agent.config import AgentContext
from react_agent.agent.contracts.dependencies import AgentDependencies
from react_agent.agent.model_execution import invoke_chat_model


def _config(*, truncate: bool = False, max_tokens: int = 64000):
    return SimpleNamespace(
        enable_history_truncation=truncate,
        max_history_tokens=max_tokens,
    )


def test_turn_segmentation_separates_completed_and_current_messages() -> None:
    messages = [
        HumanMessage(content="上一轮问题"),
        AIMessage(content="上一轮答案"),
        HumanMessage(content="当前问题"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "search",
                    "args": {"query": "当前问题"},
                }
            ],
        ),
        ToolMessage(tool_call_id="call-1", name="search", content="result"),
    ]

    segments = segment_current_turn(messages)

    assert [message.content for message in segments.completed] == [
        "上一轮问题",
        "上一轮答案",
    ]
    assert segments.current[0].content == "当前问题"
    assert latest_turn_ai_messages(messages) == [segments.current[1]]


def test_consecutive_user_messages_stay_in_one_current_turn() -> None:
    messages = [
        HumanMessage(content="上一轮"),
        AIMessage(content="已回答"),
        HumanMessage(content="当前问题第一部分"),
        HumanMessage(content="当前问题第二部分"),
    ]

    segments = segment_current_turn(messages)

    assert len(segments.completed) == 2
    assert [message.content for message in segments.current] == [
        "当前问题第一部分",
        "当前问题第二部分",
    ]


def test_history_truncation_keeps_parallel_tool_exchange_atomically() -> None:
    old_turn = [HumanMessage(content="旧问题"), AIMessage(content="旧回答")]
    recent_turn = [
        HumanMessage(content="检索问题"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call-a", "name": "search", "args": {"query": "A"}},
                {"id": "call-b", "name": "search", "args": {"query": "B"}},
            ],
        ),
        ToolMessage(tool_call_id="call-b", content="结果 B"),
        ToolMessage(tool_call_id="call-a", content="结果 A"),
        AIMessage(content="检索结论"),
    ]
    current = HumanMessage(content="追问")
    state = State(messages=old_turn + recent_turn + [current])
    max_tokens = estimate_message_tokens(recent_turn + [current])

    context = build_model_context(
        state,
        _config(truncate=True, max_tokens=max_tokens),
        system_prompt="系统",
        directive=None,
    )

    selected = context.as_list()[1:]
    assert selected == recent_turn + [current]
    assert len(validate_tool_exchanges(selected)) == 1
    assert context.diagnostics.history_truncated is True
    assert list(state.messages) == old_turn + recent_turn + [current]


def test_history_truncation_never_splits_oversized_current_exchange() -> None:
    messages = [
        HumanMessage(content="当前问题"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call-a", "name": "search", "args": {}},
                {"id": "call-b", "name": "search", "args": {}},
            ],
        ),
        ToolMessage(tool_call_id="call-a", content="结果 A"),
        ToolMessage(tool_call_id="call-b", content="结果 B"),
    ]

    context = build_model_context(
        State(messages=messages),
        _config(truncate=True, max_tokens=1),
        system_prompt="系统",
        directive=None,
    )

    assert context.as_list()[1:] == messages
    assert len(validate_tool_exchanges(context.as_list()[1:])) == 1


def test_history_truncation_does_not_skip_an_oversized_middle_turn() -> None:
    old = [HumanMessage(content="很早的问题"), AIMessage(content="很早的回答")]
    middle = [
        HumanMessage(content="中间的问题"),
        AIMessage(content="中间的长回答" * 100),
    ]
    current = HumanMessage(content="当前问题")

    context = build_model_context(
        State(messages=old + middle + [current]),
        _config(
            truncate=True,
            max_tokens=estimate_message_tokens(old + [current]),
        ),
        system_prompt="系统",
        directive=None,
    )

    assert context.as_list()[1:] == [current]


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([ToolMessage(tool_call_id="orphan", content="结果")], "孤立"),
        (
            [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "same", "name": "search", "args": {}}],
                ),
                ToolMessage(tool_call_id="same", content="第一次"),
                ToolMessage(tool_call_id="same", content="第二次"),
            ],
            "重复结果",
        ),
        (
            [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "expected", "name": "search", "args": {}}],
                ),
                ToolMessage(tool_call_id="other", content="结果"),
            ],
            "不匹配",
        ),
        (
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "same", "name": "search", "args": {}},
                        {"id": "same", "name": "search", "args": {}},
                    ],
                ),
                ToolMessage(tool_call_id="same", content="结果"),
            ],
            "重复的工具调用 ID",
        ),
    ],
)
def test_tool_exchange_validator_rejects_orphans_and_mismatches(
    messages: list, match: str
) -> None:
    with pytest.raises(ToolProtocolError, match=match):
        validate_tool_exchanges(messages)


def test_model_context_rejects_orphan_tool_result_before_provider_call() -> None:
    state = State(
        messages=[
            HumanMessage(content="问题"),
            ToolMessage(tool_call_id="orphan", content="无调用的结果"),
        ]
    )

    with pytest.raises(ToolProtocolError, match="孤立"):
        build_model_context(
            state,
            _config(),
            system_prompt="系统",
            directive=None,
        )


def test_model_context_builds_system_directive_history_and_evidence() -> None:
    tool_payload = json.dumps(
        {
            "ok": True,
            "tool": "query_internal_knowledge",
            "query": "资本开支",
            "data": {"results": []},
            "error": None,
            "meta": {},
        },
        ensure_ascii=False,
    )
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
            ToolMessage(
                tool_call_id="call-1",
                name="query_internal_knowledge",
                content=tool_payload,
            ),
        ],
        turn_evidence=[
            {
                "key": "report::1",
                "chunk_id": "report::1",
                "source": "report.pdf",
                "page": 1,
                "queries": ["资本开支"],
                "excerpt": "预计资本开支为 520 亿美元",
                "excerpt_truncated": False,
            }
        ],
    )

    context = build_model_context(
        state,
        _config(),
        system_prompt="system",
        directive="finalize",
    )

    messages = context.as_list()
    assert messages[:2] == [
        {"role": "system", "content": "system"},
        {"role": "system", "content": "finalize"},
    ]
    assert "本轮 RAG 证据索引" in messages[-1].content
    assert '"chunk_id":"report::1"' in messages[-1].content
    assert "本轮 RAG 证据索引" not in state.messages[-1].content
    assert context.diagnostics.current_turn_message_count == 3
    assert context.diagnostics.evidence_record_count == 1


def test_context_budget_falls_back_to_complete_current_turn() -> None:
    state = State(
        messages=[
            HumanMessage(content="很早的问题"),
            AIMessage(content="很早的回答"),
            HumanMessage(content="当前问题"),
            AIMessage(content="当前回答"),
        ]
    )

    context = build_model_context(
        state,
        _config(truncate=True, max_tokens=1),
        system_prompt="system",
        directive=None,
    )

    contents = [
        message.content
        for message in context.messages
        if isinstance(message, (HumanMessage, AIMessage))
    ]
    assert contents == ["当前问题", "当前回答"]
    assert context.diagnostics.history_truncated is True
    assert context.diagnostics.completed_message_count == 2


def test_context_builder_repairs_dangling_tool_protocol() -> None:
    state = State(
        messages=[
            HumanMessage(content="问题"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "missing",
                        "name": "search",
                        "args": {"query": "问题"},
                    }
                ],
            ),
        ]
    )

    context = build_model_context(
        state,
        _config(),
        system_prompt="system",
        directive=None,
    )

    assert isinstance(context.messages[-1], ToolMessage)
    assert context.messages[-1].tool_call_id == "missing"
    assert "工具调用未完成" in context.messages[-1].content


def test_conversation_summary_contract_requires_traceable_content() -> None:
    summary = ConversationSummary(
        content="用户正在比较两份资本开支指引。",
        source_message_ids=("message-1", "message-2"),
        summarized_through_message_id="message-2",
    )

    assert summary.version == 1
    with pytest.raises(ValueError, match="不能为空"):
        ConversationSummary(content="  ")


def test_token_estimator_is_offline_and_counts_tool_arguments(monkeypatch) -> None:
    original_import = builtins.__import__

    def reject_tokenizer_import(name, *args, **kwargs):
        if name == "tiktoken" or name.startswith("tiktoken."):
            raise AssertionError("上下文构造不应加载可能下载资源的 tokenizer")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_tokenizer_import)
    plain = AIMessage(content="")
    with_tool = AIMessage(
        content="",
        tool_calls=[{"id": "call-1", "name": "search", "args": {"query": "资本开支"}}],
    )

    assert estimate_message_tokens([with_tool]) > estimate_message_tokens([plain])


def test_model_context_estimates_each_original_message_once(monkeypatch) -> None:
    from react_agent.agent.context_management import budgeting

    calls: list[str] = []
    original_estimator = budgeting._estimate_text_tokens

    def counting_estimator(value: str) -> int:
        calls.append(value)
        return original_estimator(value)

    monkeypatch.setattr(budgeting, "_estimate_text_tokens", counting_estimator)
    state = State(
        messages=[HumanMessage(content="当前问题"), AIMessage(content="当前回答")]
    )

    context = build_model_context(
        state,
        _config(truncate=True, max_tokens=64000),
        system_prompt="system",
        directive=None,
    )

    assert [message.content for message in context.messages[1:]] == [
        "当前问题",
        "当前回答",
    ]
    assert calls == ["当前问题", "当前回答"]


def _budget(limit: int, *, window: int | None = None) -> ContextBudget:
    return ContextBudget(
        max_input_tokens=limit,
        reserved_completion_tokens=10,
        safety_margin_tokens=5,
        context_window_tokens=window,
    )


def test_context_budget_uses_model_window_after_completion_reservation() -> None:
    assert _budget(100, window=80).input_limit_tokens == 65
    assert _budget(100).input_limit_tokens == 100


def test_budget_report_includes_final_input_and_bound_tool_schema() -> None:
    from langchain_core.tools import StructuredTool

    def search(query: str) -> str:
        return query

    tool = StructuredTool.from_function(
        func=search, name="search", description="检索当前问题"
    )
    state = State(messages=[HumanMessage(content="当前问题")])
    context = build_model_context(
        state,
        _config(),
        system_prompt="系统规则",
        directive="当前指令",
        budget=_budget(1000),
        bound_tools=(tool,),
    )
    report = context.budget_report

    assert report is not None
    assert report.outcome == "within_budget"
    assert report.tool_schema_tokens > 0
    assert report.system_tokens > 0
    assert report.directive_tokens > 0
    assert report.estimated_input_tokens == (
        report.system_tokens
        + report.directive_tokens
        + report.tool_schema_tokens
        + report.current_user_tokens
        + report.current_turn_other_tokens
        + report.recent_history_tokens
    )
    assert report.estimated_input_tokens <= report.input_limit_tokens


@pytest.mark.parametrize(
    ("state", "system_prompt", "limit", "reason"),
    [
        (
            State(messages=[HumanMessage(content="问题")]),
            "固定系统提示" * 100,
            20,
            "FIXED_CONTEXT_TOO_LARGE",
        ),
        (
            State(messages=[HumanMessage(content="当前问题" * 100)]),
            "系统",
            20,
            "USER_INPUT_TOO_LARGE",
        ),
        (
            State(
                messages=[
                    HumanMessage(content="问题"),
                    AIMessage(
                        content="",
                        tool_calls=[{"id": "call-1", "name": "search", "args": {}}],
                    ),
                    ToolMessage(
                        tool_call_id="call-1",
                        name="search",
                        content="工具正文" * 100,
                    ),
                ],
                turn_compaction_usage={"prompt_tokens": 5},
            ),
            "系统",
            50,
            "CURRENT_TURN_CONTEXT_TOO_LARGE",
        ),
    ],
)
def test_budget_gate_classifies_irreducible_context(
    state: State, system_prompt: str, limit: int, reason: str
) -> None:
    with pytest.raises(ContextBudgetExceeded) as captured:
        build_model_context(
            state,
            _config(),
            system_prompt=system_prompt,
            directive=None,
            budget=_budget(limit),
        )

    assert captured.value.reason == reason
    assert captured.value.report.outcome == reason.lower()
    if reason == "CURRENT_TURN_CONTEXT_TOO_LARGE":
        assert len(captured.value.completed_model_messages) == 1
        assert captured.value.compaction_usage == {"prompt_tokens": 5}


def test_budget_drops_completed_history_without_mutating_state() -> None:
    old_question = HumanMessage(content="很久以前的问题" * 50)
    old_answer = AIMessage(content="很久以前的回答" * 50)
    current_question = HumanMessage(content="当前问题")
    state = State(messages=[old_question, old_answer, current_question])

    context = build_model_context(
        state,
        _config(),
        system_prompt="系统",
        directive=None,
        budget=_budget(50),
    )

    assert context.budget_report is not None
    assert context.budget_report.dropped_completed_message_count == 2
    assert [message.content for message in context.messages[1:]] == ["当前问题"]
    assert list(state.messages) == [old_question, old_answer, current_question]


def test_budget_projects_old_tool_body_before_dropping_recent_turn() -> None:
    content = json.dumps(
        {
            "ok": True,
            "tool": "web_search",
            "query": "旧问题",
            "data": {"text": "旧工具正文" * 1000},
            "meta": {},
        },
        ensure_ascii=False,
    )
    messages = [
        HumanMessage(content="旧问题"),
        AIMessage(
            content="",
            tool_calls=[{"id": "old-call", "name": "web_search", "args": {}}],
        ),
        ToolMessage(tool_call_id="old-call", name="web_search", content=content),
        AIMessage(content="旧回答"),
        HumanMessage(content="当前问题"),
    ]
    projected, _, _ = project_completed_tool_messages(messages, max_chars=512)
    limit = estimate_message_tokens([{"role": "system", "content": "系统"}] + projected)

    context = build_model_context(
        State(messages=messages),
        _config(),
        system_prompt="系统",
        directive=None,
        budget=_budget(limit),
    )

    assert [message.content for message in context.messages if isinstance(message, HumanMessage)] == [
        "旧问题",
        "当前问题",
    ]
    assert context.budget_report is not None
    assert context.budget_report.dropped_completed_message_count == 0
    assert context.budget_report.projected_tool_message_count == 1
    assert json.loads(context.messages[3].content)["meta"]["context_truncated"] is True
    assert messages[2].content == content


def test_total_budget_drops_whole_turn_with_parallel_tool_results() -> None:
    old = [
        HumanMessage(content="旧问题"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "a", "name": "search", "args": {}},
                {"id": "b", "name": "search", "args": {}},
            ],
        ),
        ToolMessage(tool_call_id="a", content="结果 A"),
        ToolMessage(tool_call_id="b", content="结果 B"),
        AIMessage(content="旧回答"),
    ]
    current = HumanMessage(content="当前问题")
    limit = estimate_message_tokens(
        [{"role": "system", "content": "系统"}, current]
    )

    context = build_model_context(
        State(messages=old + [current]),
        _config(),
        system_prompt="系统",
        directive=None,
        budget=_budget(limit),
    )

    assert context.as_list()[1:] == [current]
    assert context.budget_report is not None
    assert context.budget_report.dropped_completed_message_count == len(old)


def test_budget_counts_transient_evidence_after_projection() -> None:
    tool_message = ToolMessage(
        tool_call_id="call-1", name="query_internal_knowledge", content="结果"
    )
    state = State(
        messages=[
            HumanMessage(content="问题"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call-1", "name": "query_internal_knowledge", "args": {}}
                ],
            ),
            tool_message,
        ],
        turn_evidence=[
            {
                "key": "report::1",
                "chunk_id": "report::1",
                "source": "report.pdf",
                "page": 1,
                "queries": ["问题"],
                "excerpt": "可见原文",
                "excerpt_truncated": False,
            }
        ],
    )

    context = build_model_context(
        state,
        _config(),
        system_prompt="系统",
        directive=None,
        budget=_budget(1000),
    )

    assert context.budget_report is not None
    assert context.budget_report.evidence_tokens > 0
    assert context.budget_report.estimated_input_tokens == estimate_message_tokens(
        context.as_list()
    )
    assert "本轮 RAG 证据索引" not in tool_message.content


def test_budget_projects_large_generic_tool_output_without_mutating_state() -> None:
    content = json.dumps(
        {
            "ok": True,
            "tool": "web_search",
            "query": "问题",
            "data": {"text": "网页正文" * 1200},
            "meta": {},
        },
        ensure_ascii=False,
    )
    messages = [
        HumanMessage(content="当前问题"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "web_search", "args": {}}],
        ),
        ToolMessage(tool_call_id="call-1", name="web_search", content=content),
    ]
    projected, _, _ = project_current_tool_messages(messages, max_chars=256)
    limit = estimate_message_tokens([{"role": "system", "content": "系统"}] + projected)

    context = build_model_context(
        State(messages=messages),
        _config(),
        system_prompt="系统",
        directive=None,
        budget=_budget(limit),
    )

    visible = json.loads(context.messages[-1].content)
    assert visible["ok"] is True
    assert visible["tool"] == "web_search"
    assert visible["data"]["context_truncated"] is True
    assert visible["meta"]["original_chars"] == len(content)
    assert context.budget_report is not None
    assert context.budget_report.projected_tool_message_count == 1
    assert context.budget_report.tool_content_chars_removed > 0
    assert "tool_payload_budget" in context.budget_report.projection_reasons
    assert context.budget_report.estimated_input_tokens == estimate_message_tokens(
        context.as_list()
    )
    assert messages[-1].content == content


def test_plain_tool_projection_marks_missing_suffix() -> None:
    content = "日志正文" * 100

    projected = project_tool_content(content, 0)

    assert len(projected) < len(content)
    assert "其余正文因本次模型输入预算被截断" in projected


def test_rag_minimal_projection_never_claims_no_retrieval_hit() -> None:
    content = json.dumps(
        {
            "ok": True,
            "tool": "query_internal_knowledge",
            "query": "问题",
            "data": {
                "results": [
                    {
                        "chunk_id": "chunk-1",
                        "source": "report.pdf",
                        "page": 1,
                        "content": "证据正文" * 60,
                    }
                ]
            },
            "meta": {"retrieval_hit": True},
        },
        ensure_ascii=False,
    )

    projected = json.loads(project_tool_content(content, 0))

    assert projected["ok"] is True
    assert projected["meta"]["retrieval_hit"] is True
    assert projected["meta"]["context_truncated"] is True
    if not projected["data"]["results"]:
        assert projected["meta"]["omitted_results"] == 1
        assert "无可引用正文" in projected["meta"]["citation_warning"]


def test_budget_projects_rag_body_but_keeps_visible_provenance() -> None:
    content = json.dumps(
        {
            "ok": True,
            "tool": "query_internal_knowledge",
            "query": "资本开支",
            "data": {
                "results": [
                    {
                        "chunk_id": "report::1",
                        "source": "report.pdf",
                        "page": 1,
                        "content": "预计资本开支为 520 亿美元。" * 150,
                    }
                ]
            },
            "meta": {"has_relevant_content": True},
        },
        ensure_ascii=False,
    )
    messages = [
        HumanMessage(content="当前问题"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call-1", "name": "query_internal_knowledge", "args": {}}
            ],
        ),
        ToolMessage(
            tool_call_id="call-1",
            name="query_internal_knowledge",
            content=content,
        ),
    ]
    projected, _, _ = project_current_tool_messages(messages, max_chars=512)
    limit = estimate_message_tokens([{"role": "system", "content": "系统"}] + projected)

    context = build_model_context(
        State(messages=messages),
        _config(),
        system_prompt="系统",
        directive=None,
        budget=_budget(limit),
    )

    visible = json.loads(context.messages[-1].content)
    result = visible["data"]["results"][0]
    assert visible["ok"] is True
    assert visible["meta"]["context_truncated"] is True
    assert result["chunk_id"] == "report::1"
    assert result["source"] == "report.pdf"
    assert result["page"] == 1
    assert result["content_truncated"] is True
    assert messages[-1].content == content


def test_budget_projects_evidence_index_and_reports_omitted_records() -> None:
    records = [
        {
            "chunk_id": f"report::{index}",
            "source": f"report-{index}.pdf",
            "page": index + 1,
            "queries": ["问题"],
            "excerpt": "证据正文" * 50,
            "excerpt_truncated": False,
        }
        for index in range(18)
    ]
    messages = [
        HumanMessage(content="当前问题"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "search", "args": {}}],
        ),
        ToolMessage(tool_call_id="call-1", name="search", content="简短结果"),
    ]
    index = render_evidence_index(records, max_records=1, max_excerpt_chars=0)
    limited = attach_evidence_index(messages, index)
    limit = estimate_message_tokens([{"role": "system", "content": "系统"}] + limited)
    state = State(messages=messages, turn_evidence=records)

    context = build_model_context(
        state,
        _config(),
        system_prompt="系统",
        directive=None,
        budget=_budget(limit),
    )

    report = context.budget_report
    assert report is not None
    assert report.evidence_records_omitted_by_budget == 17
    assert report.evidence_excerpt_chars_removed > 0
    assert "evidence_record_budget" in report.projection_reasons
    assert "另有 17 条候选片段未纳入索引" in context.messages[-1].content
    assert '"chunk_id":"report::0"' in context.messages[-1].content
    assert state.turn_evidence == records


@pytest.mark.asyncio
async def test_budget_gate_runs_before_model_provider_resolution() -> None:
    resolved = False

    def model_provider():
        nonlocal resolved
        resolved = True
        raise AssertionError("超限输入不得解析或调用模型")

    dependencies = AgentDependencies(
        config=AgentContext(
            system_prompt="系统",
            enable_tools=False,
            max_history_tokens=20,
            max_input_tokens=20,
        ),
        model_provider=model_provider,
        tools=(),
    )
    state = State(messages=[HumanMessage(content="当前问题" * 100)])

    with pytest.raises(ContextBudgetExceeded, match="USER_INPUT_TOO_LARGE"):
        await invoke_chat_model(
            state,
            dependencies,
            None,
            allow_tools=False,
            directive=None,
        )
    assert resolved is False
