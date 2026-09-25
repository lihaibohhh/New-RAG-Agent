"""MCP 工具调用的最小结构化追踪。"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger("mcp_service.tools")


@dataclass
class ToolCallTrace:
    tool: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _started: float = field(default_factory=time.perf_counter, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    @classmethod
    def start(cls, tool: str) -> "ToolCallTrace":
        trace = cls(tool=tool)
        logger.info(
            "mcp_tool event=start request_id=%s tool=%s",
            trace.request_id,
            trace.tool,
        )
        return trace

    def finish(
        self,
        *,
        stage: str,
        status: str,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("同一 MCP ToolCallTrace 只能结束一次")
        self._finished = True
        elapsed_ms = round((time.perf_counter() - self._started) * 1000, 2)
        logger.info(
            "mcp_tool event=end request_id=%s tool=%s stage=%s "
            "status=%s elapsed_ms=%.2f error_type=%s",
            self.request_id,
            self.tool,
            stage,
            status,
            elapsed_ms,
            error_type or "",
        )
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "tool": self.tool,
            "stage": stage,
            "status": status,
            "elapsed_ms": elapsed_ms,
        }
        if error_type:
            payload["error_type"] = error_type
        return payload


__all__ = ["ToolCallTrace"]
