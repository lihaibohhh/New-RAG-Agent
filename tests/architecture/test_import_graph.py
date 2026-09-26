"""AST 导入图工具自身的回归测试。"""

from __future__ import annotations

from pathlib import Path

from tests.architecture.import_graph import collect_imports, is_module_or_child


def test_collect_imports_ignores_comments_and_string_literals(tmp_path: Path) -> None:
    package_root = tmp_path / "sample"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "module.py").write_text(
        '"""import knowledge.ingestion"""\n'
        "# from knowledge.server import app\n"
        "import httpx\n",
        encoding="utf-8",
    )

    references = collect_imports(package_root, project_root=tmp_path)

    assert [(item.source_module, item.imported_module) for item in references] == [
        ("sample.module", "httpx")
    ]


def test_collect_imports_resolves_relative_and_from_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "sample"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "module.py").write_text(
        "from .submodule import Service\n" "from knowledge import ingestion\n",
        encoding="utf-8",
    )

    references = collect_imports(package_root, project_root=tmp_path)
    candidates = {
        candidate
        for reference in references
        for candidate in reference.candidate_modules
    }

    assert "sample.submodule" in candidates
    assert "knowledge.ingestion" in candidates


def test_module_prefix_matching_respects_package_boundaries() -> None:
    assert is_module_or_child("knowledge.runtime.config", "knowledge.runtime")
    assert not is_module_or_child("knowledge.runtime_ports", "knowledge.runtime")
