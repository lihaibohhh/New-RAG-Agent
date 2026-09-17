"""Health endpoints distinguish process liveness from persistence readiness."""

from __future__ import annotations

import httpx
import pytest

import api.main as api_main
from react_agent.runtime.container import ApplicationStatus


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "effective", "expected_code"),
    [
        ("sqlite", "sqlite", 200),
        ("sqlite", "memory", 503),
        ("postgres", "postgres", 200),
        ("postgres", "memory", 503),
        ("memory", "memory", 200),
    ],
)
async def test_readiness_reports_checkpoint_backend_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    effective: str,
    expected_code: int,
) -> None:
    async def fake_status() -> ApplicationStatus:
        return ApplicationStatus(
            agent_initialized=True,
            requested_checkpoint_backend=requested,
            checkpoint_backend=effective,
        )

    monkeypatch.setattr(api_main, "get_runtime_status", fake_status)
    transport = httpx.ASGITransport(app=api_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ready_response = await client.get("/api/v1/health/ready")
        live_response = await client.get("/api/v1/health")

    assert ready_response.status_code == expected_code
    if expected_code == 503:
        assert ready_response.headers["content-type"].startswith(
            "application/problem+json"
        )
        assert ready_response.json()["status"] == 503
        assert requested in ready_response.json()["detail"]
        assert effective in ready_response.json()["detail"]
    else:
        assert ready_response.json()["checkpoint_backend"] == effective
    assert live_response.status_code == 200
    assert live_response.json()["checkpoint_backend"] == effective


@pytest.mark.asyncio
async def test_readiness_fails_when_agent_is_not_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_status() -> ApplicationStatus:
        return ApplicationStatus(
            agent_initialized=False,
            requested_checkpoint_backend="memory",
            checkpoint_backend="memory",
        )

    monkeypatch.setattr(api_main, "get_runtime_status", fake_status)
    transport = httpx.ASGITransport(app=api_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health/ready")

    assert response.status_code == 503
