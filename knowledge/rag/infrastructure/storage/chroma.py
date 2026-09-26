"""RAG Chroma 只读访问前的目录边界检查。"""

from __future__ import annotations

import os
from pathlib import Path


def validate_chroma_directory(chroma_dir: str) -> Path:
    """拒绝错误目录和空 Chroma 实例，并返回规范化目录。"""
    path = Path(chroma_dir).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"Chroma 目录不存在：{path}")
    database_file = path / "chroma.sqlite3"
    if not database_file.is_file():
        raise RuntimeError(f"Chroma 数据库文件不存在：{database_file}")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        raise PermissionError(f"Chroma 目录不可读写：{path}")
    return path


__all__ = ["validate_chroma_directory"]
