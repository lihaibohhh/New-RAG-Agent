"""Configuration contract for the remote Knowledge Service client."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeClientConfig:
    """Connection settings required by a remote Knowledge Service runtime."""

    base_url: str
    api_key: str = ""
    timeout: float = 150.0

    def __post_init__(self) -> None:
        normalized = self.base_url.strip().rstrip("/")
        if not normalized:
            raise ValueError("Knowledge Service base_url 不能为空")
        if self.timeout < 1.0:
            raise ValueError("Knowledge Service timeout 必须大于或等于 1 秒")
        object.__setattr__(self, "base_url", normalized)
        object.__setattr__(self, "api_key", self.api_key.strip())


__all__ = ["KnowledgeClientConfig"]
