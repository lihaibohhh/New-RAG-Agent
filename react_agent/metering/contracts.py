"""模型用量与费用估算的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True)
class CostEstimate:
    """一次模型调用的可审计费用估算。"""

    amount: float | None
    currency: str
    status: Literal["estimated", "unpriced", "invalid_usage"]
    billing_model: str | None
    tariff: Literal["peak", "off_peak"] | None
    rate_card_version: str


CostEstimator = Callable[[str, dict[str, int]], CostEstimate]


__all__ = ["CostEstimate", "CostEstimator"]
