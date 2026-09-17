from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from react_agent.metering.pricing import (
    estimate_model_cost_cny,
    load_pricing_catalog,
)
from react_agent.observability.display import (
    SessionUsageTracker,
    format_cost_cny,
    format_usage_for_user,
)
from react_agent.metering.turn import extract_usage


PRICING_PATH = (
    Path(__file__).resolve().parents[1]
    / "react_agent"
    / "configuration"
    / "deepseek_pricing.yaml"
)


@pytest.fixture(scope="module")
def catalog():
    return load_pricing_catalog(PRICING_PATH)


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("deepseek-flash", "deepseek-flash"),
        ("deepseek/deepseek-v4-flash", "deepseek-flash"),
        ("deepseek-chat", "deepseek-flash"),
    ],
)
def test_model_aliases_resolve_to_current_flash(catalog, model_name, expected) -> None:
    assert catalog.canonical_model(model_name) == expected


def test_off_peak_cost_uses_cny_rate_card(catalog) -> None:
    estimate = estimate_model_cost_cny(
        model_name="deepseek-v4-flash",
        usage={
            "cache_hit_tokens": 1_000_000,
            "cache_miss_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        },
        catalog=catalog,
        at=datetime(2026, 9, 19, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert estimate.status == "estimated"
    assert estimate.tariff == "off_peak"
    assert estimate.amount == pytest.approx(5.02)
    assert estimate.currency == "CNY"


def test_peak_cost_uses_weekday_time_windows(catalog) -> None:
    estimate = estimate_model_cost_cny(
        model_name="deepseek-flash",
        usage={
            "cache_hit_tokens": 1_000_000,
            "cache_miss_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        },
        catalog=catalog,
        at=datetime(2026, 9, 18, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert estimate.status == "estimated"
    assert estimate.tariff == "peak"
    assert estimate.amount == pytest.approx(10.04)


def test_unknown_model_is_unpriced_instead_of_free(catalog) -> None:
    estimate = estimate_model_cost_cny(
        model_name="deepseek-future-model",
        usage={"cache_miss_tokens": 1000, "completion_tokens": 100},
        catalog=catalog,
    )

    assert estimate.status == "unpriced"
    assert estimate.amount is None


def test_negative_usage_is_explicitly_invalid(catalog) -> None:
    estimate = estimate_model_cost_cny(
        model_name="deepseek-flash",
        usage={"cache_miss_tokens": -1},
        catalog=catalog,
    )

    assert estimate.status == "invalid_usage"
    assert estimate.amount is None


def test_usage_delta_and_display_use_native_cny() -> None:
    result = {
        "estimated_cost_cny": 0.015,
        "unpriced_model_count": 0,
        "total_tokens": 100,
        "llm_call_count": 1,
        "pricing_model": "deepseek-flash",
        "pricing_tariff": "off_peak",
        "pricing_rate_card_version": "2026-08-17",
        "tool_runs": [],
    }
    usage = extract_usage(result)

    assert usage["estimated_cost_cny"] == pytest.approx(0.015)
    assert format_usage_for_user(usage, 500)["本轮成本"] == "¥0.02"
    assert format_cost_cny(0) == "¥0.0000"
    assert format_cost_cny(0, unpriced_count=1) == "费用不可用（1 次未计价）"


def test_session_tracker_accumulates_cny_and_unpriced_calls() -> None:
    tracker = SessionUsageTracker()
    tracker.record_turn(
        {
            "estimated_cost_cny": 0.015,
            "unpriced_model_count": 1,
            "total_tokens": 100,
            "llm_call_count": 1,
            "tool_runs_count": 0,
        },
        latency_ms=10,
    )

    assert tracker.total_cost_cny == pytest.approx(0.015)
    assert tracker.total_unpriced_model_calls == 1
