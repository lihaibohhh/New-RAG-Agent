from __future__ import annotations

import json
import ast
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from react_agent.agent.tool_calls import extract_tool_call_ids
from react_agent.agent.tool_policy import (
    bound_tool_payload,
    count_successful_rag_calls_in_current_turn,
)
from react_agent.tooling.results import tool_error, tool_success
from react_agent.tooling.retry import with_retry


def test_tool_call_ids_cover_valid_invalid_and_provider_payloads() -> None:
    valid = AIMessage(
        content="",
        tool_calls=[{"id": "valid-1", "name": "search", "args": {}}],
        invalid_tool_calls=[
            {"id": "invalid-1", "name": "search", "args": "{"}
        ],
    )
    provider_payload = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {"id": "raw-1", "type": "function", "function": {}}
            ]
        },
    )

    assert extract_tool_call_ids(valid) == ["valid-1", "invalid-1"]
    assert extract_tool_call_ids(provider_payload) == ["raw-1"]


def test_rag_call_policy_counts_only_successes_in_current_turn() -> None:
    messages = [
        HumanMessage(content="上一轮"),
        ToolMessage(
            name="query_internal_knowledge",
            tool_call_id="old",
            content=json.dumps(tool_success(tool_name="rag", query="old", data={})),
        ),
        HumanMessage(content="当前轮"),
        ToolMessage(
            name="query_internal_knowledge",
            tool_call_id="failed",
            content=json.dumps(
                tool_error(tool_name="rag", query="new", message="timeout")
            ),
        ),
        ToolMessage(
            name="query_internal_knowledge",
            tool_call_id="success",
            content=json.dumps(tool_success(tool_name="rag", query="new", data={})),
        ),
    ]

    assert count_successful_rag_calls_in_current_turn(messages) == 1


def test_tool_payload_bounding_preserves_envelope_and_traceability() -> None:
    payload = tool_success(
        tool_name="rag",
        query="query",
        data={
            "results": [
                {"content": "a" * 80, "source_file": "one.pdf", "source_page": 1},
                {"content": "b" * 80, "source_file": "two.pdf", "source_page": 2},
            ]
        },
    )

    bounded = json.loads(bound_tool_payload(json.dumps(payload), max_chars=150))

    assert bounded["ok"] is True
    assert bounded["tool"] == "rag"
    assert bounded["meta"]["truncated"] is True
    assert bounded["meta"]["total_results"] == 2


@pytest.mark.asyncio
async def test_tool_retry_uses_public_result_contract() -> None:
    attempts = 0

    @with_retry(tool_name="test", max_retries=1, base_delay=0)
    async def flaky(query: str) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary")
        return tool_success(tool_name="test", query=query, data={"done": True})

    payload = json.loads(await flaky("hello"))

    assert attempts == 2
    assert payload["ok"] is True
    assert payload["meta"]["attempt"] == 1


def _python_sources(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def _imports_from(source_path: Path, module_prefix: str) -> list[str]:
    imports: list[str] = []
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            module_prefix
        ):
            imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith(module_prefix)
            )
    return imports


def test_utils_package_is_fully_removed() -> None:
    package_root = Path(__file__).parent.parent / "react_agent"

    assert not (package_root / "utils").exists()
    for source_path in _python_sources(package_root):
        assert "react_agent.utils" not in source_path.read_text(encoding="utf-8")


def test_internal_symbols_are_not_imported_across_project_modules() -> None:
    project_root = Path(__file__).parent.parent
    source_roots = (
        project_root / "react_agent",
        project_root / "api",
        project_root / "knowledge_service",
        project_root / "eval",
        project_root / "scripts",
    )
    violations: list[str] = []
    for source_root in source_roots:
        for source_path in _python_sources(source_root):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if not (node.module or "").startswith("react_agent."):
                    continue
                for imported in node.names:
                    if imported.name.startswith("_"):
                        violations.append(
                            f"{source_path.relative_to(project_root)} imports "
                            f"{node.module}.{imported.name}"
                        )
    assert violations == []


def test_agent_does_not_depend_on_outbound_adapter_implementations() -> None:
    package_root = Path(__file__).parent.parent / "react_agent"
    forbidden_prefixes = (
        "react_agent.tools",
        "react_agent.models",
        "react_agent.infrastructure",
    )
    violations: dict[str, list[str]] = {}
    for path in _python_sources(package_root / "agent"):
        imports = [
            imported
            for prefix in forbidden_prefixes
            for imported in _imports_from(path, prefix)
        ]
        if imports:
            violations[str(path.relative_to(package_root))] = imports
    assert violations == {}


def test_infrastructure_does_not_depend_on_agent() -> None:
    package_root = Path(__file__).parent.parent / "react_agent"
    infrastructure_roots = (
        package_root / "infrastructure",
        package_root / "models",
        package_root / "observability",
        package_root / "rag" / "infrastructure",
        package_root / "conversations" / "infrastructure",
    )
    violations: dict[str, list[str]] = {}
    for root in infrastructure_roots:
        for path in _python_sources(root):
            imports = _imports_from(path, "react_agent.agent")
            if imports:
                violations[str(path.relative_to(package_root))] = imports
    assert violations == {}
