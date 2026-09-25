"""Knowledge Service 自身的轻量环境配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} 必须是布尔值："
        "true/false、1/0、yes/no 或 on/off"
    )


@dataclass(frozen=True)
class KnowledgeServiceSettings:
    """HTTP 服务边界配置；本地 Runtime 配置由 server.runtime 组装。"""

    api_key: str = ""
    require_api_key: bool = False
    ingestion_root: Path = Path("./FinancialResearchReportData")
    warmup_on_start: bool = False
    evaluation_api_enabled: bool = False

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
            evaluation_api_enabled=_env_bool(
                "KNOWLEDGE_SERVICE_EVALUATION_API_ENABLED"
            ),
        )


__all__ = ["KnowledgeServiceSettings"]
