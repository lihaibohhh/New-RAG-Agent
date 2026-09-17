"""模型构造、调用编排、计量与观测之间的依赖边界。"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from react_agent.observability.usage import log_usage


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "react_agent"


def _imports(directory: Path) -> set[str]:
    names: set[str] = set()
    for path in directory.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
    return names


def test_package_imports_follow_the_metering_boundary() -> None:
    model_imports = _imports(PACKAGE_ROOT / "models")
    metering_imports = _imports(PACKAGE_ROOT / "metering")
    observation_imports = _imports(PACKAGE_ROOT / "observability")
    modeling_imports = _imports(PACKAGE_ROOT / "agent" / "modeling")

    assert not any(
        name.startswith(("react_agent.agent", "react_agent.metering", "react_agent.observability"))
        for name in model_imports
    )
    assert not any(
        name.startswith(("react_agent.agent", "react_agent.models", "react_agent.observability"))
        for name in metering_imports
    )
    assert not any(
        name.startswith(("react_agent.agent", "react_agent.models", "react_agent.metering.pricing"))
        for name in observation_imports
    )
    assert not any(
        name.startswith(("react_agent.models", "react_agent.observability", "react_agent.metering.pricing"))
        for name in modeling_imports
    )


def test_usage_logger_only_writes_precomputed_usage(tmp_path: Path) -> None:
    usage = {"total_tokens": 17, "estimated_cost_cny": 0.05}
    log_path = tmp_path / "usage.jsonl"

    assert log_usage(
        usage,
        username="tester",
        thread_id="thread-1",
        question="问题",
        latency_ms=12.3,
        log_path=log_path,
    ) is None

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["total_tokens"] == 17
    assert record["estimated_cost_cny"] == 0.05
    assert usage == {"total_tokens": 17, "estimated_cost_cny": 0.05}
