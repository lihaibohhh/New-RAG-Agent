"""RAG 各层共享的来源标识规则。"""
from __future__ import annotations

from pathlib import Path


def source_identity(file_path: str, source_root: str | None) -> tuple[str, str | None]:
    """返回稳定的相对来源名和可选行业目录。"""
    path = Path(file_path).resolve()
    relative = Path(path.name)
    if source_root:
        try:
            relative = path.relative_to(Path(source_root).resolve())
        except ValueError:
            pass

    source_file = relative.as_posix()
    industry = relative.parts[0] if len(relative.parts) > 1 else None
    return source_file, industry


__all__ = ["source_identity"]
