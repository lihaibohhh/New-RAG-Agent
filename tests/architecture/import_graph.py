"""使用 Python AST 收集源码中的静态模块依赖。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportReference:
    """一条源码导入及其可用于依赖匹配的候选模块。"""

    source_path: Path
    source_module: str
    imported_module: str
    imported_names: tuple[str, ...]
    line: int

    @property
    def candidate_modules(self) -> tuple[str, ...]:
        candidates = [self.imported_module]
        candidates.extend(
            f"{self.imported_module}.{name}"
            for name in self.imported_names
            if name != "*"
        )
        return tuple(candidates)


def is_module_or_child(module: str, prefix: str) -> bool:
    """判断模块是否等于给定前缀或位于其包内。"""
    return module == prefix or module.startswith(f"{prefix}.")


def collect_imports(
    package_root: Path, *, project_root: Path
) -> tuple[ImportReference, ...]:
    """解析一个包内所有 Python 文件，返回确定性的静态导入集合。"""
    references: list[ImportReference] = []
    for source_path in sorted(package_root.rglob("*.py")):
        source_module = _module_name(source_path, project_root=project_root)
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                references.extend(
                    ImportReference(
                        source_path=source_path,
                        source_module=source_module,
                        imported_module=alias.name,
                        imported_names=(),
                        line=node.lineno,
                    )
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                references.append(
                    ImportReference(
                        source_path=source_path,
                        source_module=source_module,
                        imported_module=_resolve_from_module(
                            node,
                            source_path=source_path,
                            source_module=source_module,
                        ),
                        imported_names=tuple(alias.name for alias in node.names),
                        line=node.lineno,
                    )
                )
    return tuple(
        sorted(
            references,
            key=lambda item: (
                str(item.source_path),
                item.line,
                item.imported_module,
            ),
        )
    )


def _module_name(source_path: Path, *, project_root: Path) -> str:
    relative = source_path.relative_to(project_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_from_module(
    node: ast.ImportFrom,
    *,
    source_path: Path,
    source_module: str,
) -> str:
    if node.level == 0:
        return node.module or ""

    if source_path.name == "__init__.py":
        package_parts = source_module.split(".")
    else:
        package_parts = source_module.split(".")[:-1]
    levels_up = node.level - 1
    if levels_up:
        if levels_up > len(package_parts):
            raise ValueError(f"无法解析越界相对导入：{source_path}:{node.lineno}")
        package_parts = package_parts[:-levels_up]
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(package_parts)


__all__ = ["ImportReference", "collect_imports", "is_module_or_child"]
