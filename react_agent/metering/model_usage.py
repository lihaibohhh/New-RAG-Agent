"""跨 Agent 与 API 的模型用量归一化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from react_agent.metering.contracts import CostEstimate, CostEstimator


@dataclass(frozen=True)
class ModelCallMetering:
    """一次模型调用的统一用量、真实模型名和费用估算。"""

    usage: dict[str, int]
    model_name: str
    cost: CostEstimate


def extract_model_usage(message: AIMessage) -> dict[str, int]:
    """把非流式和流式 LangChain 用量统一为 Agent 的计费字段。"""
    raw: dict[str, Any] = {}
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict) and token_usage:
            raw = token_usage

    if raw:
        cache_hit = int(raw.get("prompt_cache_hit_tokens") or 0)
        missing = raw.get("prompt_cache_miss_tokens")
        cache_miss = (
            int(missing)
            if missing is not None
            else max(0, int(raw.get("prompt_tokens") or 0) - cache_hit)
        )
        completion = int(raw.get("completion_tokens") or 0)
        details = raw.get("completion_tokens_details")
        reasoning = (
            int(details.get("reasoning_tokens") or 0)
            if isinstance(details, dict)
            else 0
        )
    else:
        usage_metadata = getattr(message, "usage_metadata", None) or {}
        input_details = usage_metadata.get("input_token_details") or {}
        cache_hit = int(input_details.get("cache_read") or 0)
        total_input = int(usage_metadata.get("input_tokens") or 0)
        cache_miss = max(0, total_input - cache_hit)
        completion = int(usage_metadata.get("output_tokens") or 0)
        output_details = usage_metadata.get("output_token_details") or {}
        reasoning = int(output_details.get("reasoning_tokens") or 0)

    prompt = cache_hit + cache_miss
    return {
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "prompt_tokens": prompt,
        "total_tokens": prompt + completion,
    }


def meter_model_call(
    message: AIMessage,
    *,
    model_ref: str,
    cost_estimator: CostEstimator | None,
) -> ModelCallMetering:
    """Agent 与 API 共用的单次模型计量入口。"""
    usage = extract_model_usage(message)
    metadata = getattr(message, "response_metadata", None) or {}
    model_name = (
        metadata.get("model_name")
        or metadata.get("model")
        or model_ref.split("/")[-1]
    )
    name = str(model_name)
    return ModelCallMetering(
        usage=usage,
        model_name=name,
        cost=(
            cost_estimator(name, usage)
            if cost_estimator is not None
            else CostEstimate(
                amount=None,
                currency="CNY",
                status="unpriced",
                billing_model=None,
                tariff=None,
                rate_card_version="unconfigured",
            )
        ),
    )


__all__ = [
    "ModelCallMetering",
    "extract_model_usage",
    "meter_model_call",
]
