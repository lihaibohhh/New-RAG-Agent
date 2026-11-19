from __future__ import annotations

import json

import httpx
import pytest

from api.client import AgentAPIError, ResearchAgentClient


@pytest.mark.asyncio
async def test_invoke_sends_auth_and_decodes_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/invoke"
        assert request.headers["X-API-Key"] == "test-key"
        assert json.loads(request.content) == {
            "message": "问题",
            "session_id": "session",
        }
        return httpx.Response(
            200,
            json={
                "content": "回答",
                "session_id": "session",
                "thread_id": "user:test:session",
            },
        )

    async with ResearchAgentClient(
        base_url="http://test",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.invoke("问题", "session")

    assert result["content"] == "回答"


@pytest.mark.asyncio
async def test_stream_events_decodes_sse_frames() -> None:
    body = (
        'event: tool_call\ndata: {"name":"search","args":{"query":"测试"}}\n\n'
        'event: token\ndata: {"delta":"答"}\n\n'
        'event: done\ndata: {"total_tokens":3}\n\n'
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=body,
        )

    async with ResearchAgentClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        events = [
            event
            async for event in client.stream_events("问题", session_id="session")
        ]

    assert [event.type for event in events] == ["tool_call", "token", "done"]
    assert events[0].data["args"]["query"] == "测试"
    assert events[1].data["delta"] == "答"


@pytest.mark.asyncio
async def test_stream_text_raises_for_in_stream_error() -> None:
    body = (
        'event: token\ndata: {"delta":"部分"}\n\n'
        "event: error\n"
        'data: {"status":504,"detail":"流超时","request_id":"req-1"}\n\n'
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    async with ResearchAgentClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        stream = client.stream_text("问题")
        assert await stream.__anext__() == "部分"
        with pytest.raises(AgentAPIError) as captured:
            await stream.__anext__()

    assert captured.value.status_code == 504
    assert captured.value.request_id == "req-1"


@pytest.mark.asyncio
async def test_problem_response_becomes_agent_api_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"X-Request-Id": "req-2"},
            json={"detail": "请求过多"},
        )

    async with ResearchAgentClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AgentAPIError) as captured:
            await client.invoke("问题")

    assert captured.value.status_code == 429
    assert captured.value.detail == "请求过多"
    assert captured.value.request_id == "req-2"
