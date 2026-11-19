"""Lightweight asynchronous client for the Research Agent HTTP API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx


@dataclass(frozen=True)
class StreamEvent:
    """One decoded Server-Sent Event from the Agent API."""

    type: str
    data: dict[str, Any]


class AgentAPIError(RuntimeError):
    """An HTTP-level or in-stream error returned by the Agent API."""

    def __init__(
        self,
        detail: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.detail = detail
        self.status_code = status_code
        self.request_id = request_id
        parts = [detail]
        if status_code is not None:
            parts.insert(0, f"HTTP {status_code}")
        if request_id:
            parts.append(f"request_id={request_id}")
        super().__init__(" | ".join(parts))


class ResearchAgentClient:
    """Small async wrapper around the Agent REST and SSE endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        stream_read_timeout: float | None = 660.0,
        connect_timeout: float = 10.0,
        write_timeout: float = 30.0,
        pool_timeout: float = 10.0,
        trust_env: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url:
            raise ValueError("base_url 不能为空")

        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        self._http = httpx.AsyncClient(
            base_url=normalized_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=connect_timeout,
                write=write_timeout,
                read=stream_read_timeout,
                pool=pool_timeout,
            ),
            trust_env=trust_env,
            transport=transport,
        )

    async def __aenter__(self) -> ResearchAgentClient:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: Any,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()

    async def invoke(
        self,
        message: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run a non-streaming turn and return the decoded response object."""
        response = await self._http.post(
            "/api/v1/chat/invoke",
            json=_chat_payload(message, session_id),
        )
        if response.is_error:
            raise _api_error(response)
        payload = response.json()
        if not isinstance(payload, dict):
            raise AgentAPIError("Agent API 返回的不是 JSON 对象")
        return payload

    async def stream_events(
        self,
        message: str,
        session_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield decoded token, tool, usage, error, and done events."""
        async with self._http.stream(
            "POST",
            "/api/v1/chat/stream",
            headers={"Accept": "text/event-stream"},
            json=_chat_payload(message, session_id),
        ) as response:
            if response.is_error:
                await response.aread()
                raise _api_error(response)

            async for event in _decode_sse(response.aiter_lines()):
                yield event

    async def stream_text(
        self,
        message: str,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield only visible answer text and raise on an in-stream error."""
        async for event in self.stream_events(message, session_id=session_id):
            if event.type == "token":
                delta = event.data.get("delta")
                if isinstance(delta, str) and delta:
                    yield delta
            elif event.type == "error":
                raise AgentAPIError(
                    str(event.data.get("detail") or "Agent 流式调用失败"),
                    status_code=_optional_int(event.data.get("status")),
                    request_id=_optional_string(event.data.get("request_id")),
                )


def _chat_payload(message: str, session_id: str | None) -> dict[str, Any]:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message 不能为空")
    return {"message": message, "session_id": session_id}


async def _decode_sse(lines: AsyncIterator[str]) -> AsyncIterator[StreamEvent]:
    event_type = "message"
    data_lines: list[str] = []

    async for line in lines:
        if line == "":
            if data_lines:
                yield _build_event(event_type, data_lines)
            event_type = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_type = value or "message"
        elif field == "data":
            data_lines.append(value)

    if data_lines:
        yield _build_event(event_type, data_lines)


def _build_event(event_type: str, data_lines: list[str]) -> StreamEvent:
    raw_data = "\n".join(data_lines)
    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise AgentAPIError(f"SSE data 不是合法 JSON: {raw_data[:200]}") from exc
    if not isinstance(payload, dict):
        raise AgentAPIError("SSE data 必须是 JSON 对象")
    return StreamEvent(type=event_type, data=payload)


def _api_error(response: httpx.Response) -> AgentAPIError:
    detail = response.text or response.reason_phrase
    request_id = response.headers.get("X-Request-Id")
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = None
    if isinstance(payload, dict):
        detail = str(payload.get("detail") or payload.get("title") or detail)
        request_id = _optional_string(payload.get("request_id")) or request_id
    return AgentAPIError(
        detail,
        status_code=response.status_code,
        request_id=request_id,
    )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


__all__ = ["AgentAPIError", "ResearchAgentClient", "StreamEvent"]
