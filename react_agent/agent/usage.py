"""Agent 模型用量归一化与费用计算。"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


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
        cache_miss = int(raw.get("prompt_cache_miss_tokens") or 0)
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


def estimate_model_cost_usd(
    *,
    model_name: str,
    usage: dict[str, int],
    price_table: dict[str, Any],
) -> float:
    """根据每百万 token 价格估算一次模型调用费用。"""
    price = price_table.get(model_name) or {}
    if not price:
        return 0.0
    return (
        usage["cache_hit_tokens"] / 1_000_000 * price["cache_hit"]
        + usage["cache_miss_tokens"] / 1_000_000 * price["cache_miss"]
        + usage["completion_tokens"] / 1_000_000 * price["output"]
    )


__all__ = ["estimate_model_cost_usd", "extract_model_usage"]
