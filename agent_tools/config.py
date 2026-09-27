"""Agent 工具适配器拥有的非敏感配置模型。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


_APP_ROOT = Path(__file__).resolve().parent.parent


def _resolve_app_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _APP_ROOT / path
    return str(path.resolve())


class RagToolConfig(BaseModel):
    """Agent 调用远程 Knowledge Service 时的适配器策略。"""

    max_retries: int = Field(default=2, ge=0)
    client_timeout: float = Field(default=150.0, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: Any) -> dict[str, Any]:
        payload = dict(values or {})
        timeout = os.getenv("KNOWLEDGE_SERVICE_TIMEOUT", "").strip()
        if timeout:
            payload["client_timeout"] = timeout
        return payload


class ExcelToolConfig(BaseModel):
    mode: Literal["timestamp", "overwrite", "append"] = "timestamp"
    keep_backup: bool = False


class SearchToolConfig(BaseModel):
    max_retries: int = Field(default=2, ge=0)
    timeout: int = Field(default=15, gt=0)
    max_search_results: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: Any) -> dict[str, Any]:
        payload = dict(values or {})
        env_mapping = {
            "max_retries": "SEARCH_MAX_RETRIES",
            "timeout": "SEARCH_TIMEOUT",
            "max_search_results": "MAX_SEARCH_RESULTS",
        }
        for field_name, env_name in env_mapping.items():
            value = os.getenv(env_name, "").strip()
            if value:
                payload[field_name] = value
        return payload


class SqlToolConfig(BaseModel):
    DB_PATH: str = str(_APP_ROOT / "data_sql" / "financials.db")

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: Any) -> dict[str, Any]:
        payload = dict(values or {})
        for env_name in ("SQL_DB_PATH", "DB_PATH"):
            value = os.getenv(env_name, "").strip()
            if value:
                payload["DB_PATH"] = value
                break
        return payload

    @model_validator(mode="after")
    def _resolve_path(self) -> "SqlToolConfig":
        object.__setattr__(self, "DB_PATH", _resolve_app_path(self.DB_PATH))
        return self


class AgentToolsConfig(BaseModel):
    rag: RagToolConfig = Field(default_factory=RagToolConfig)
    excel: ExcelToolConfig = Field(default_factory=ExcelToolConfig)
    search: SearchToolConfig = Field(default_factory=SearchToolConfig)
    sql_store: SqlToolConfig = Field(default_factory=SqlToolConfig)


__all__ = [
    "AgentToolsConfig",
    "ExcelToolConfig",
    "RagToolConfig",
    "SearchToolConfig",
    "SqlToolConfig",
]
