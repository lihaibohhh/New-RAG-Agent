"""Knowledge 包的模块依赖方向守卫。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.architecture.import_graph import (
    ImportReference,
    collect_imports,
    is_module_or_child,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"

FORBIDDEN_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "knowledge.rag": (
        "knowledge.ingestion",
        "knowledge.runtime",
        "knowledge.runtime_ports",
        "knowledge.server",
        "knowledge.client",
        "knowledge.transport",
        "react_agent",
        "mcp_service",
        "mcp_server",
        "api",
        "fastapi",
    ),
    "knowledge.ingestion": (
        "knowledge.rag",
        "knowledge.runtime",
        "knowledge.runtime_ports",
        "knowledge.server",
        "knowledge.client",
        "knowledge.transport",
        "react_agent",
        "mcp_service",
        "mcp_server",
        "api",
        "fastapi",
    ),
    "knowledge.foundation": (
        "knowledge.rag",
        "knowledge.ingestion",
        "knowledge.runtime",
        "knowledge.runtime_ports",
        "knowledge.server",
        "knowledge.client",
        "knowledge.transport",
        "react_agent",
        "mcp_service",
        "mcp_server",
        "api",
        "fastapi",
    ),
    "knowledge.transport": (
        "knowledge.rag",
        "knowledge.ingestion",
        "knowledge.runtime",
        "knowledge.server",
        "knowledge.client",
        "knowledge.foundation",
        "react_agent",
        "mcp_service",
        "mcp_server",
        "api",
        "fastapi",
        "httpx",
    ),
    "knowledge.client": (
        "knowledge.rag",
        "knowledge.ingestion",
        "knowledge.runtime",
        "knowledge.server",
        "knowledge.foundation",
        "react_agent",
        "mcp_service",
        "mcp_server",
        "api",
    ),
    "knowledge.runtime": (
        "knowledge.server",
        "knowledge.client",
        "knowledge.transport",
        "react_agent",
        "mcp_service",
        "mcp_server",
        "api",
        "fastapi",
    ),
    "knowledge.server": (
        "knowledge.foundation",
        "knowledge.client",
        "react_agent",
        "mcp_service",
        "mcp_server",
        "api",
    ),
}

RUNTIME_MODULE_ALLOWLIST: dict[str, dict[str, tuple[str, ...]]] = {
    "knowledge.runtime.config": {
        "knowledge.rag": ("knowledge.rag.config",),
        "knowledge.ingestion": ("knowledge.ingestion.config",),
    },
    "knowledge.runtime.container": {
        "knowledge.rag": ("knowledge.rag.runtime",),
        "knowledge.ingestion": ("knowledge.ingestion.runtime",),
    },
}

SERVER_MODULE_ALLOWLIST: dict[str, dict[str, tuple[str, ...]]] = {
    "knowledge.server.runtime": {
        "knowledge.rag": ("knowledge.rag.config",),
        "knowledge.ingestion": ("knowledge.ingestion.config",),
    },
}


@pytest.fixture(scope="module")
def knowledge_imports() -> tuple[ImportReference, ...]:
    return collect_imports(KNOWLEDGE_ROOT, project_root=PROJECT_ROOT)


def test_knowledge_packages_respect_forbidden_dependency_rules(
    knowledge_imports: tuple[ImportReference, ...],
) -> None:
    violations: list[str] = []
    for reference in knowledge_imports:
        for source_prefix, forbidden_prefixes in FORBIDDEN_DEPENDENCIES.items():
            if not is_module_or_child(reference.source_module, source_prefix):
                continue
            forbidden = _first_matching_prefix(
                reference,
                forbidden_prefixes,
            )
            if forbidden is not None:
                violations.append(_violation(reference, forbidden))

    assert not violations, "发现禁止的 Knowledge 依赖：\n" + "\n".join(violations)


def test_top_runtime_only_imports_module_composition_boundaries(
    knowledge_imports: tuple[ImportReference, ...],
) -> None:
    violations = _find_imports_outside_scoped_allowlist(
        knowledge_imports,
        source_prefix="knowledge.runtime",
        source_allowlist=RUNTIME_MODULE_ALLOWLIST,
    )

    assert not violations, "顶层 Runtime 越过了模块组合边界：\n" + "\n".join(violations)


def test_server_only_imports_module_configuration_boundaries(
    knowledge_imports: tuple[ImportReference, ...],
) -> None:
    violations = _find_imports_outside_scoped_allowlist(
        knowledge_imports,
        source_prefix="knowledge.server",
        source_allowlist=SERVER_MODULE_ALLOWLIST,
    )

    assert not violations, "Knowledge Server 越过了模块配置边界：\n" + "\n".join(
        violations
    )


def _first_matching_prefix(
    reference: ImportReference,
    prefixes: tuple[str, ...],
) -> str | None:
    return next(
        (
            prefix
            for prefix in prefixes
            if any(
                is_module_or_child(candidate, prefix)
                for candidate in reference.candidate_modules
            )
        ),
        None,
    )


def _find_imports_outside_scoped_allowlist(
    references: tuple[ImportReference, ...],
    *,
    source_prefix: str,
    source_allowlist: dict[str, dict[str, tuple[str, ...]]],
) -> list[str]:
    monitored_prefixes = {
        module_prefix
        for module_rules in source_allowlist.values()
        for module_prefix in module_rules
    }
    violations: list[str] = []
    for reference in references:
        if not is_module_or_child(reference.source_module, source_prefix):
            continue
        module_rules = source_allowlist.get(reference.source_module, {})
        for module_prefix in monitored_prefixes:
            matching_targets = tuple(
                target
                for target in reference.candidate_modules
                if is_module_or_child(target, module_prefix)
            )
            if not matching_targets:
                continue
            allowed_prefixes = module_rules.get(module_prefix, ())
            if not any(
                is_module_or_child(target, allowed)
                for target in matching_targets
                for allowed in allowed_prefixes
            ):
                rule = (
                    f"仅允许 {', '.join(allowed_prefixes)}"
                    if allowed_prefixes
                    else f"{reference.source_module} 不允许直接依赖 {module_prefix}"
                )
                violations.append(_violation(reference, rule))
    return violations


def _violation(reference: ImportReference, rule: str) -> str:
    relative_path = reference.source_path.relative_to(PROJECT_ROOT)
    imported = ", ".join(reference.candidate_modules)
    return (
        f"- {relative_path}:{reference.line} "
        f"{reference.source_module} -> {imported}（{rule}）"
    )
