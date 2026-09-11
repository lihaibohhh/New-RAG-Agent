"""Knowledge Service 自身的轻量环境配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KnowledgeServiceSettings:
    """只包含服务边界配置；RAG 内部配置仍由项目 settings 管理。"""

    api_key: str = ""
    require_api_key: bool = False
    ingestion_root: Path = Path("./FinancialResearchReportData")
    warmup_on_start: bool = False

    @classmethod
    def from_env(cls) -> "KnowledgeServiceSettings":
        return cls(
            api_key=os.getenv("KNOWLEDGE_SERVICE_API_KEY", "").strip(),
            require_api_key=_env_bool("KNOWLEDGE_SERVICE_REQUIRE_API_KEY"),
            ingestion_root=Path(
                os.getenv(
                    "KNOWLEDGE_SERVICE_INGESTION_ROOT",
                    "./FinancialResearchReportData",
                )
            ).expanduser(),
            warmup_on_start=_env_bool("KNOWLEDGE_SERVICE_WARMUP_ON_START"),
        )


__all__ = ["KnowledgeServiceSettings"]
