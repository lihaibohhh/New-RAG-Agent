"""DeepSeek 模型价格配置、别名解析与人民币费用估算。"""

from __future__ import annotations

import logging
import os
from datetime import datetime, time, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field, model_validator

from react_agent.metering.contracts import CostEstimate


logger = logging.getLogger(__name__)
_DEFAULT_PRICING_PATH = (
    Path(__file__).resolve().parents[1] / "configuration" / "deepseek_pricing.yaml"
)


class TokenRates(BaseModel):
    """每百万 Token 的人民币价格。"""

    cache_hit_input: Decimal = Field(ge=0)
    cache_miss_input: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)


class TariffRates(BaseModel):
    off_peak: TokenRates
    peak: TokenRates


class PeakRange(BaseModel):
    start: time
    end: time

    @model_validator(mode="after")
    def _validate_range(self) -> "PeakRange":
        if self.start >= self.end:
            raise ValueError("峰时区间 start 必须早于 end")
        return self


class PeakWindows(BaseModel):
    weekdays: set[int] = Field(default_factory=set)
    ranges: list[PeakRange] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_weekdays(self) -> "PeakWindows":
        if any(day < 1 or day > 7 for day in self.weekdays):
            raise ValueError("weekdays 必须使用 ISO 1..7")
        return self


class ModelPricing(BaseModel):
    display_version: str
    aliases: list[str] = Field(default_factory=list)
    effective_from: datetime
    rates: TariffRates
    peak_windows: PeakWindows


class DeepSeekPricingCatalog(BaseModel):
    schema_version: int = Field(ge=1)
    rate_card_version: str
    currency: Literal["CNY"] = "CNY"
    unit_tokens: int = Field(default=1_000_000, gt=0)
    timezone: str = "Asia/Shanghai"
    models: dict[str, ModelPricing]

    @model_validator(mode="after")
    def _validate_aliases(self) -> "DeepSeekPricingCatalog":
        seen: dict[str, str] = {}
        for canonical, pricing in self.models.items():
            for name in (canonical, *pricing.aliases):
                normalized = _normalize_model_name(name)
                owner = seen.get(normalized)
                if owner is not None and owner != canonical:
                    raise ValueError(f"模型别名 {name!r} 同时指向 {owner!r} 和 {canonical!r}")
                seen[normalized] = canonical
        ZoneInfo(self.timezone)
        return self

    def canonical_model(self, model_name: str) -> str | None:
        normalized = _normalize_model_name(model_name)
        for canonical, pricing in self.models.items():
            candidates = (canonical, *pricing.aliases)
            if any(_normalize_model_name(item) == normalized for item in candidates):
                return canonical
        return None


def _normalize_model_name(model_name: str) -> str:
    """同时接受 `provider/model` 和 API 返回的裸模型名。"""
    normalized = str(model_name or "").strip().lower()
    if "/" in normalized:
        provider, bare_name = normalized.split("/", 1)
        if provider in {"deepseek", "ds"}:
            normalized = bare_name
    return normalized


def pricing_path() -> Path:
    configured = os.getenv("DEEPSEEK_PRICING_PATH", "").strip()
    return Path(configured).expanduser().resolve() if configured else _DEFAULT_PRICING_PATH


@lru_cache(maxsize=8)
def load_pricing_catalog(path: str | Path | None = None) -> DeepSeekPricingCatalog:
    resolved = Path(path).resolve() if path is not None else pricing_path()
    with resolved.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return DeepSeekPricingCatalog.model_validate(payload)


def _tariff_at(
    model: ModelPricing,
    catalog: DeepSeekPricingCatalog,
    at: datetime,
) -> Literal["peak", "off_peak"]:
    instant = at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)
    local = instant.astimezone(ZoneInfo(catalog.timezone))
    if local.isoweekday() not in model.peak_windows.weekdays:
        return "off_peak"
    local_time = local.timetz().replace(tzinfo=None)
    if any(window.start <= local_time < window.end for window in model.peak_windows.ranges):
        return "peak"
    return "off_peak"


def estimate_model_cost_cny(
    *,
    model_name: str,
    usage: dict[str, int],
    catalog: DeepSeekPricingCatalog | None = None,
    at: datetime | None = None,
) -> CostEstimate:
    """按调用时间和 API usage 估算费用；未知模型显式返回 unpriced。"""
    selected = catalog or load_pricing_catalog()
    canonical = selected.canonical_model(model_name)
    if canonical is None:
        logger.warning("[pricing] 未配置模型价格: model=%s", model_name)
        return CostEstimate(
            amount=None,
            currency=selected.currency,
            status="unpriced",
            billing_model=None,
            tariff=None,
            rate_card_version=selected.rate_card_version,
        )

    calculated_at = at or datetime.now(timezone.utc)
    model = selected.models[canonical]
    effective_from = model.effective_from
    if effective_from.tzinfo is None:
        effective_from = effective_from.replace(tzinfo=ZoneInfo(selected.timezone))
    if calculated_at.tzinfo is None:
        calculated_at = calculated_at.replace(tzinfo=timezone.utc)
    if calculated_at < effective_from:
        logger.warning(
            "[pricing] 调用时间早于价格卡生效时间: model=%s at=%s effective_from=%s",
            canonical,
            calculated_at.isoformat(),
            effective_from.isoformat(),
        )
        return CostEstimate(
            amount=None,
            currency=selected.currency,
            status="unpriced",
            billing_model=canonical,
            tariff=None,
            rate_card_version=selected.rate_card_version,
        )

    token_counts = {
        key: int(usage.get(key, 0))
        for key in (
            "cache_hit_tokens",
            "cache_miss_tokens",
            "completion_tokens",
        )
    }
    if any(value < 0 for value in token_counts.values()):
        logger.warning("[pricing] Token 用量不能为负数: model=%s", canonical)
        return CostEstimate(
            amount=None,
            currency=selected.currency,
            status="invalid_usage",
            billing_model=canonical,
            tariff=None,
            rate_card_version=selected.rate_card_version,
        )

    tariff = _tariff_at(model, selected, calculated_at)
    rates = model.rates.peak if tariff == "peak" else model.rates.off_peak
    amount = (
        Decimal(token_counts["cache_hit_tokens"]) * rates.cache_hit_input
        + Decimal(token_counts["cache_miss_tokens"]) * rates.cache_miss_input
        + Decimal(token_counts["completion_tokens"]) * rates.output
    ) / Decimal(selected.unit_tokens)
    return CostEstimate(
        amount=float(amount),
        currency=selected.currency,
        status="estimated",
        billing_model=canonical,
        tariff=tariff,
        rate_card_version=selected.rate_card_version,
    )


def estimate_configured_model_cost(
    model_name: str,
    usage: dict[str, int],
) -> CostEstimate:
    """使用应用统一价格表估算模型费用，匹配 Agent 注入契约。"""
    return estimate_model_cost_cny(model_name=model_name, usage=usage)


__all__ = [
    "DeepSeekPricingCatalog",
    "estimate_configured_model_cost",
    "estimate_model_cost_cny",
    "load_pricing_catalog",
    "pricing_path",
]
