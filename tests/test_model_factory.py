from __future__ import annotations

import json

import httpx
from langchain_core.messages import HumanMessage

from react_agent.models import factory
from react_agent.models.factory import (
    ChatModelSettings,
    bind_output_token_limit,
    output_token_limit_kwargs,
)


def _config(max_tokens: int = 8192) -> ChatModelSettings:
    return ChatModelSettings(
        temperature=0.2,
        max_tokens=max_tokens,
        timeout=10,
        retries=0,
    )


def test_deepseek_request_uses_provider_max_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        factory,
        "_get_secret",
        lambda attribute, *_args: (
            "https://api.deepseek.test" if attribute == "DEEPSEEK_BASE_URL" else "fake"
        ),
    )

    model = factory._build_deepseek("deepseek-flash", _config())
    payload = model._get_request_payload("test")

    assert payload["extra_body"] == {"max_tokens": 8192}
    assert "max_completion_tokens" not in payload


def test_deepseek_http_body_places_max_tokens_at_top_level(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    monkeypatch.setattr(
        factory,
        "_get_secret",
        lambda attribute, *_args: (
            "https://api.deepseek.test" if attribute == "DEEPSEEK_BASE_URL" else "fake"
        ),
    )
    model = factory._build_deepseek("deepseek-flash", _config())
    model.root_client = model.root_client.copy(
        http_client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    model.client = model.root_client.chat.completions

    response = model.invoke("test")

    assert response.content == "ok"
    assert captured["max_tokens"] == 8192
    assert "max_completion_tokens" not in captured
    assert "extra_body" not in captured

    captured.clear()
    compacting_model = bind_output_token_limit(
        model,
        model_ref="deepseek/deepseek-flash",
        max_tokens=768,
    )
    compacting_model.invoke("test")

    assert captured["max_tokens"] == 768
    assert "max_completion_tokens" not in captured


def test_openai_request_uses_max_completion_tokens(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    model = factory._build_openai("gpt-4.1-mini", _config())
    payload = model._get_request_payload("test")

    assert payload["max_completion_tokens"] == 8192
    assert "extra_body" not in payload


def test_anthropic_request_uses_max_tokens(monkeypatch) -> None:
    monkeypatch.setattr(factory.settings.secrets, "ANTHROPIC_API_KEY", "fake")

    model = factory._build_anthropic("claude-test", _config())
    payload = model._get_request_payload([HumanMessage(content="test")])

    assert payload["max_tokens"] == 8192
    assert "max_completion_tokens" not in payload


def test_per_call_limit_mapping_is_provider_aware() -> None:
    assert output_token_limit_kwargs("deepseek/deepseek-flash", 768) == {
        "extra_body": {"max_tokens": 768}
    }
    assert output_token_limit_kwargs("openai/gpt-4.1-mini", 768) == {
        "max_completion_tokens": 768
    }
    assert output_token_limit_kwargs("anthropic/claude-test", 768) == {
        "max_tokens": 768
    }
