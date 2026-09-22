"""API 用量、费用与预算的离线回归。"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from api.models import ChatRequest
from api.errors import AppError
from api.routes.v1 import chat
from react_agent.agent.context_management import (
    BudgetReport,
    ContextBudgetExceeded,
    latest_turn_ai_messages,
)
from react_agent.metering.contracts import CostEstimate
from react_agent.metering.model_usage import (
    extract_model_usage,
    meter_model_call,
)


def _message(model: str, prompt: int, completion: int, *, content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        response_metadata={
            "model_name": model,
            "token_usage": {
                "prompt_tokens": prompt,
                "prompt_cache_hit_tokens": 0,
                "completion_tokens": completion,
            },
        },
    )


def _estimate(model_name: str, usage: dict[str, int]) -> CostEstimate:
    known = model_name != "unknown-model"
    return CostEstimate(
        amount=usage["total_tokens"] * 0.01 if known else None,
        currency="CNY",
        status="estimated" if known else "unpriced",
        billing_model=model_name if known else None,
        tariff="off_peak" if known else None,
        rate_card_version="test",
    )


def _state() -> dict[str, Any]:
    return {
        "ttft_ms": None,
        "total_tokens": 0,
        "total_cost": 0.0,
        "unpriced_model_count": 0,
        "models": {},
    }


def test_unexpected_boundary_node_result_reaches_stream_client() -> None:
    frame = chat._map_event(
        {
            "event": "on_chain_end",
            "name": "call_model",
            "data": {
                "output": {
                    "termination_reason": "UNEXPECTED_RECURSION_BOUNDARY",
                    "messages": [AIMessage(content="执行边界提示")],
                }
            },
        },
        _state(),
        None,
        time.perf_counter(),
    )
    assert frame is not None
    assert frame.startswith("event: token\n")
    assert json.loads(frame.split("data: ", 1)[1])["delta"] == "执行边界提示"


def _compaction_usage() -> dict[str, Any]:
    return {
        "model_name": "known-model",
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "total_tokens": 6,
        "cost_cny": 0.06,
        "priced": True,
    }


def test_provider_prompt_total_fills_missing_cache_miss() -> None:
    usage = extract_model_usage(_message("known-model", 8, 2))
    assert usage["prompt_tokens"] == 8
    assert usage["cache_miss_tokens"] == 8
    assert usage["total_tokens"] == 10


def test_unconfigured_cost_policy_is_owned_by_metering() -> None:
    metered = meter_model_call(
        _message("known-model", 8, 2),
        model_ref="test/known-model",
        cost_estimator=None,
    )
    assert metered.usage["total_tokens"] == 10
    assert metered.cost.status == "unpriced"
    assert metered.cost.rate_card_version == "unconfigured"


def test_latest_turn_includes_tool_planning_and_final_model_calls() -> None:
    messages = [
        HumanMessage(content="上一轮"),
        _message("known-model", 100, 10, content="旧回答"),
        HumanMessage(content="本轮"),
        _message("known-model", 8, 2),
        _message("known-model", 5, 3, content="最终回答"),
    ]
    assert sum(
        extract_model_usage(message)["total_tokens"]
        for message in latest_turn_ai_messages(messages)
    ) == 18


def test_stream_usage_serializes_cost_and_unknown_price(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "estimate_configured_model_cost", _estimate)

    class Agent:
        model_ref = "test/known-model"

    state = _state()
    first = chat._map_event(
        {"event": "on_chat_model_end", "data": {"output": _message("known-model", 8, 2)}},
        state,
        Agent(),
        time.perf_counter(),
    )
    second = chat._map_event(
        {"event": "on_chat_model_end", "data": {"output": _message("unknown-model", 5, 3)}},
        state,
        Agent(),
        time.perf_counter(),
    )

    assert json.loads(first.split("data: ", 1)[1])["cost"] == pytest.approx(0.10)
    unknown = json.loads(second.split("data: ", 1)[1])
    assert unknown["cost"] is None
    assert unknown["cost_status"] == "unpriced"
    assert state["total_tokens"] == 18
    assert state["total_cost"] == pytest.approx(0.10)
    assert state["unpriced_model_count"] == 1


def test_stream_hides_compaction_tokens_but_counts_its_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat, "estimate_configured_model_cost", _estimate)

    class Agent:
        model_ref = "test/known-model"

    state = _state()
    hidden = chat._map_event(
        {
            "event": "on_chat_model_stream",
            "tags": ["context_compaction"],
            "data": {"chunk": AIMessage(content="内部摘要")},
        },
        state,
        Agent(),
        time.perf_counter(),
    )
    usage = chat._map_event(
        {
            "event": "on_chat_model_end",
            "tags": ["context_compaction"],
            "data": {"output": _message("known-model", 5, 1)},
        },
        state,
        Agent(),
        time.perf_counter(),
    )
    assert hidden is None
    assert usage is not None and usage.startswith("event: usage")
    assert state["total_tokens"] == 6


@pytest.mark.asyncio
async def test_stream_records_all_tokens_and_emits_done(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "estimate_configured_model_cost", _estimate)
    recorded: list[tuple[str, int]] = []

    async def record(bucket_key: str, tokens: int) -> None:
        recorded.append((bucket_key, tokens))

    monkeypatch.setattr(chat, "record_token_usage", record)

    class Request:
        async def is_disconnected(self) -> bool:
            return False

    class Agent:
        model_ref = "test/known-model"

        async def stream_events(self, messages: list[Any], thread_id: str):
            assert thread_id == "user:test:session"
            yield {
                "event": "on_chat_model_end",
                "data": {"output": _message("known-model", 8, 2)},
            }
            yield {
                "event": "on_chat_model_end",
                "data": {"output": _message("unknown-model", 5, 3)},
            }

    frames = [
        frame
        async for frame in chat._stream_v1_generator(
            Request(), Agent(), "问题", "user:test:session", "session", "bucket"
        )
    ]
    assert [frame.split("\n", 1)[0] for frame in frames] == [
        "event: usage",
        "event: usage",
        "event: done",
    ]
    done = json.loads(frames[-1].split("data: ", 1)[1])
    assert done["total_tokens"] == 18
    assert done["total_cost"] == pytest.approx(0.10)
    assert done["currency"] == "CNY"
    assert done["unpriced_model_count"] == 1
    assert recorded == [("bucket", 18)]


@pytest.mark.asyncio
async def test_invoke_records_every_model_call_in_latest_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat, "estimate_configured_model_cost", _estimate)
    recorded: list[tuple[str, int]] = []

    async def no_limit(*args: Any) -> None:
        return None

    async def record(bucket_key: str, tokens: int) -> None:
        recorded.append((bucket_key, tokens))

    monkeypatch.setattr(chat, "check_rate_limit", no_limit)
    monkeypatch.setattr(chat, "check_token_budget", no_limit)
    monkeypatch.setattr(chat, "record_token_usage", record)

    class Agent:
        model_ref = "test/known-model"

        async def invoke(self, messages: list[Any], thread_id: str) -> dict[str, Any]:
            assert thread_id == "user:api-key:session"
            return {
                "messages": [
                    HumanMessage(content="上一轮"),
                    _message("known-model", 100, 10, content="旧回答"),
                    HumanMessage(content="本轮"),
                    _message("known-model", 8, 2),
                    _message("known-model", 5, 3, content="最终回答"),
                ]
            }

    response = await chat.chat_invoke(
        ChatRequest(message="本轮", session_id="session"),
        None,
        Agent(),
        "api-key",
    )
    assert response.content == "最终回答"
    assert recorded == [("api-key", 18)]


@pytest.mark.asyncio
async def test_invoke_adds_compaction_call_to_token_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat, "estimate_configured_model_cost", _estimate)
    recorded: list[int] = []

    async def no_limit(*args: Any) -> None:
        return None

    async def record(bucket_key: str, tokens: int) -> None:
        recorded.append(tokens)

    monkeypatch.setattr(chat, "check_rate_limit", no_limit)
    monkeypatch.setattr(chat, "check_token_budget", no_limit)
    monkeypatch.setattr(chat, "record_token_usage", record)

    class Agent:
        model_ref = "test/known-model"

        async def invoke(self, messages: list[Any], thread_id: str) -> dict[str, Any]:
            return {
                "messages": [HumanMessage(content="问题"), _message("known-model", 8, 2, content="回答")],
                "turn_compaction_usage": _compaction_usage(),
            }

    response = await chat.chat_invoke(
        ChatRequest(message="问题", session_id="session"), None, Agent(), "api-key"
    )
    assert response.content == "回答"
    assert recorded == [16]


def _budget_error(
    reason: str,
    *,
    completed_model_messages: tuple[AIMessage, ...] = (),
    compaction_usage: dict[str, Any] | None = None,
) -> ContextBudgetExceeded:
    return ContextBudgetExceeded(
        reason,
        BudgetReport(
            input_limit_tokens=20,
            estimated_input_tokens=100,
            system_tokens=5,
            directive_tokens=0,
            tool_schema_tokens=0,
            current_user_tokens=95,
            current_turn_other_tokens=0,
            evidence_tokens=0,
            recent_history_tokens=0,
            reserved_completion_tokens=10,
            safety_margin_tokens=5,
            context_window_tokens=None,
            outcome=reason.lower(),
        ),
        completed_model_messages=completed_model_messages,
        compaction_usage=compaction_usage,
    )


@pytest.mark.asyncio
async def test_stream_budget_error_keeps_error_done_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def record(bucket_key: str, tokens: int) -> None:
        assert tokens == 0

    monkeypatch.setattr(chat, "record_token_usage", record)

    class Request:
        async def is_disconnected(self) -> bool:
            return False

    class Agent:
        async def stream_events(self, messages: list[Any], thread_id: str):
            raise _budget_error("USER_INPUT_TOO_LARGE")
            yield  # pragma: no cover

    frames = [
        frame
        async for frame in chat._stream_v1_generator(
            Request(), Agent(), "问题", "user:test:session", "session", "bucket"
        )
    ]

    assert [frame.split("\n", 1)[0] for frame in frames] == [
        "event: error",
        "event: done",
    ]
    error = json.loads(frames[0].split("data: ", 1)[1])
    assert error["status"] == 413
    assert "问题超过" in error["detail"]


@pytest.mark.asyncio
async def test_stream_graph_step_error_keeps_error_done_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[tuple[str, int]] = []

    async def record(bucket_key: str, tokens: int) -> None:
        recorded.append((bucket_key, tokens))

    monkeypatch.setattr(chat, "record_token_usage", record)

    class Request:
        async def is_disconnected(self) -> bool:
            return False

    class Agent:
        model_ref = "test/known-model"

        async def stream_events(self, messages: list[Any], thread_id: str):
            yield {
                "event": "on_chat_model_end",
                "data": {"output": _message("known-model", 8, 2)},
            }
            raise GraphRecursionError("test limit")

    frames = [
        frame
        async for frame in chat._stream_v1_generator(
            Request(), Agent(), "问题", "user:test:session", "session", "bucket"
        )
    ]
    assert [frame.split("\n", 1)[0] for frame in frames] == [
        "event: usage",
        "event: error",
        "event: done",
    ]
    error = json.loads(frames[1].split("data: ", 1)[1])
    assert error["title"] == "Agent Step Budget Exceeded"
    assert error["status"] == 500
    assert recorded == [("bucket", 10)]


@pytest.mark.asyncio
async def test_invoke_graph_step_error_uses_problem_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[tuple[str, int]] = []

    async def no_limit(*args: Any) -> None:
        return None

    async def record(bucket_key: str, tokens: int) -> None:
        recorded.append((bucket_key, tokens))

    monkeypatch.setattr(chat, "check_rate_limit", no_limit)
    monkeypatch.setattr(chat, "check_token_budget", no_limit)
    monkeypatch.setattr(chat, "record_token_usage", record)

    class Agent:
        model_ref = "test/known-model"

        async def invoke(self, messages: list[Any], thread_id: str) -> dict[str, Any]:
            error = GraphRecursionError("test limit")
            error.completed_model_messages = (_message("known-model", 8, 2),)
            raise error

    with pytest.raises(AppError) as captured:
        await chat.chat_invoke(
            ChatRequest(message="问题", session_id="session"),
            None,
            Agent(),
            "api-key",
        )

    assert captured.value.status == 500
    assert captured.value.title == "Agent Step Budget Exceeded"
    assert recorded == [("api-key", 10)]


@pytest.mark.asyncio
async def test_invoke_budget_error_uses_problem_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_limit(*args: Any) -> None:
        return None

    monkeypatch.setattr(chat, "check_rate_limit", no_limit)
    monkeypatch.setattr(chat, "check_token_budget", no_limit)
    monkeypatch.setattr(chat, "record_token_usage", no_limit)

    class Agent:
        async def invoke(self, messages: list[Any], thread_id: str) -> dict[str, Any]:
            raise _budget_error("USER_INPUT_TOO_LARGE")

    with pytest.raises(AppError) as captured:
        await chat.chat_invoke(
            ChatRequest(message="问题", session_id="session"),
            None,
            Agent(),
            "api-key",
        )

    assert captured.value.status == 413
    assert "问题超过" in captured.value.detail


@pytest.mark.asyncio
async def test_invoke_budget_error_records_prior_model_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat, "estimate_configured_model_cost", _estimate)
    recorded: list[tuple[str, int]] = []

    async def no_limit(*args: Any) -> None:
        return None

    async def record(bucket_key: str, tokens: int) -> None:
        recorded.append((bucket_key, tokens))

    monkeypatch.setattr(chat, "check_rate_limit", no_limit)
    monkeypatch.setattr(chat, "check_token_budget", no_limit)
    monkeypatch.setattr(chat, "record_token_usage", record)

    class Agent:
        model_ref = "test/known-model"

        async def invoke(self, messages: list[Any], thread_id: str) -> dict[str, Any]:
            raise _budget_error(
                "CURRENT_TURN_CONTEXT_TOO_LARGE",
                completed_model_messages=(_message("known-model", 8, 2),),
            )

    with pytest.raises(AppError) as captured:
        await chat.chat_invoke(
            ChatRequest(message="问题", session_id="session"),
            None,
            Agent(),
            "api-key",
        )

    assert captured.value.status == 413
    assert recorded == [("api-key", 10)]


@pytest.mark.asyncio
async def test_invoke_budget_error_also_records_compaction_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat, "estimate_configured_model_cost", _estimate)
    recorded: list[int] = []

    async def no_limit(*args: Any) -> None:
        return None

    async def record(bucket_key: str, tokens: int) -> None:
        recorded.append(tokens)

    monkeypatch.setattr(chat, "check_rate_limit", no_limit)
    monkeypatch.setattr(chat, "check_token_budget", no_limit)
    monkeypatch.setattr(chat, "record_token_usage", record)

    class Agent:
        model_ref = "test/known-model"

        async def invoke(self, messages: list[Any], thread_id: str) -> dict[str, Any]:
            raise _budget_error(
                "CURRENT_TURN_CONTEXT_TOO_LARGE",
                completed_model_messages=(_message("known-model", 8, 2),),
                compaction_usage=_compaction_usage(),
            )

    with pytest.raises(AppError) as captured:
        await chat.chat_invoke(
            ChatRequest(message="问题", session_id="session"), None, Agent(), "api-key"
        )
    assert captured.value.status == 413
    assert recorded == [16]
