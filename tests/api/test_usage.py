"""API 用量、费用与预算的离线回归。"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from api.models import ChatRequest
from api.routes.v1 import chat
from react_agent.agent.modeling.history import latest_turn_ai_messages
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
