"""当前轮工具计量不得依赖有界历史轨迹的长度。"""

from react_agent.metering.turn import extract_cumulative_snapshot, extract_usage


def test_tool_usage_survives_bounded_history_rollover() -> None:
    previous = extract_cumulative_snapshot(
        {
            "tool_run_count": 48,
            "tool_runs": [{"tool": "historical"}] * 48,
        }
    )
    current_runs = [
        {"tool": "query_internal_knowledge", "query": "rag-1"},
        {"tool": "query_internal_knowledge", "query": "rag-2"},
        {"tool": "search", "query": "search-1"},
        {"tool": "search", "query": "search-2"},
    ]
    result = {
        "tool_run_count": 52,
        "tool_runs": ([{"tool": "historical"}] * 46) + current_runs,
        "turn_tool_runs": current_runs,
    }

    usage = extract_usage(result, previous)

    assert len(result["tool_runs"]) == 50
    assert usage["tool_runs_count"] == 4
    assert result["turn_tool_runs"] == current_runs


def test_legacy_checkpoint_starts_new_tool_counter_from_zero() -> None:
    previous = extract_cumulative_snapshot(
        {"tool_runs": [{"tool": "historical"}] * 50}
    )
    result = {
        "tool_run_count": 2,
        "tool_runs": [{"tool": "historical"}] * 50,
        "turn_tool_runs": [{"tool": "search"}, {"tool": "search"}],
    }

    assert previous["tool_run_count"] == 0
    assert extract_usage(result, previous)["tool_runs_count"] == 2
