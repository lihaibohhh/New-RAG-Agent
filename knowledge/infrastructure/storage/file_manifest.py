"""基于 JSON 文件的增量建库指纹清单。"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

class JsonIngestionManifestAdapter:
    """持久化文件哈希，并通过同目录原子替换避免半写文件。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"建库指纹清单必须是 JSON 对象：{self.path}")
        return {str(key): str(value) for key, value in payload.items()}

    def save(self, record: dict[str, str]) -> None:
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(record, stream, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
                temp_path = Path(stream.name)
            os.replace(temp_path, target)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()


__all__ = ["JsonIngestionManifestAdapter"]
