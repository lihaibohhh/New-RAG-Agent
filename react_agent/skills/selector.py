"""第一版内置 Skill 的确定性选择规则。"""

from __future__ import annotations

import re


_RESEARCH_INTENT_SIGNALS = (
    "分析",
    "研究",
    "现状",
    "趋势",
    "展望",
    "格局",
    "规模",
    "供需",
    "驱动",
    "风险",
    "机会",
    "影响",
    "景气",
)
_MARKET_SUBJECT_SIGNALS = (
    "行业",
    "市场",
    "产业",
    "赛道",
    "板块",
    "领域",
    "产业链",
    "市场规模",
    "竞争格局",
    "供给",
    "需求",
    "价格周期",
    "渗透率",
)
_COMPANY_FINANCIAL_PATTERN = re.compile(
    r"(公司|股份|集团).{0,12}(财务|营收|收入|净利润|毛利率|估值)|"
    r"(财务|营收|净利润|毛利率|估值).{0,12}(公司|股份|集团)"
)


def select_builtin_skill_name(text: str) -> tuple[str, str] | None:
    """返回匹配的内置 Skill 名称和可观测选择原因。"""
    normalized = " ".join(str(text or "").strip().split()).casefold()
    if not normalized:
        return None

    intent_signal = next(
        (signal for signal in _RESEARCH_INTENT_SIGNALS if signal in normalized),
        None,
    )
    subject_signal = next(
        (signal for signal in _MARKET_SUBJECT_SIGNALS if signal in normalized),
        None,
    )
    if intent_signal and subject_signal and not _COMPANY_FINANCIAL_PATTERN.search(normalized):
        return (
            "industry-market-research",
            f"research_intent={intent_signal}; market_subject={subject_signal}",
        )
    return None


__all__ = ["select_builtin_skill_name"]
