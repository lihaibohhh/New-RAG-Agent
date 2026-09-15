"""Agent 时间表示规则。"""
from __future__ import annotations

from datetime import datetime


def now_iso_in_timezone(timezone_name: str) -> str:
    """返回指定 IANA 时区的 ISO 时间；无效时区降级为本地时间。"""
    try:
        from zoneinfo import ZoneInfo

        timezone = ZoneInfo(timezone_name)
    except Exception:
        timezone = None
    return datetime.now(tz=timezone).isoformat()


__all__ = ["now_iso_in_timezone"]
