"""最终回答的可回答性行为裁判。"""

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from eval.pipeline.metrics.answerability import evaluate_answer_behavior

logger = logging.getLogger(__name__)


class AnswerBehaviorVerdict(BaseModel):
    """裁判只识别回答行为，不读取数据集标签。"""

    case_id: str
    behavior: Literal["supported_answer", "abstention", "unsupported_answer"]
    rationale: str = Field(description="一句话说明分类依据")


class AnswerBehaviorBatch(BaseModel):
    items: list[AnswerBehaviorVerdict] = Field(default_factory=list)


_CLEAR_ABSTENTION_RE = re.compile(
    r"(资料|参考资料|上下文|知识库|所给信息).{0,12}"
    r"(不足|未提供|没有|无法|不能).{0,12}(确定|判断|回答|得出|查到)|"
    r"无法根据.{0,20}(确定|判断|回答|得出)|信息不足",
    re.IGNORECASE,
)
_SPECIFIC_NUMBER_RE = re.compile(
    r"(?<!\w)\d+(?:\.\d+)?\s*(?:%|万|亿|元|年|月|日|倍)"
)


def fallback_answer_behavior(answer: str) -> str:
    """裁判失败时只识别明确且没有具体数字的短拒答。"""
    text = str(answer or "").strip()
    if not text:
        return "empty_response"
    if (
        len(text) <= 180
        and _CLEAR_ABSTENTION_RE.search(text)
        and not _SPECIFIC_NUMBER_RE.search(text)
    ):
        return "abstention"
    return "judge_error"


def _structured_llm(llm: Any):
    structured = llm.with_structured_output(
        AnswerBehaviorBatch,
        method="function_calling",
    )
    api_base = str(getattr(llm, "openai_api_base", "") or "").casefold()
    model_name = str(getattr(llm, "model_name", "") or "").casefold()
    if "deepseek" in api_base or "deepseek" in model_name:
        structured = structured.bind(extra_body={"thinking": {"type": "disabled"}})
    return structured


def _payload(record: dict, index: int) -> dict[str, Any]:
    return {
        "case_id": str(record.get("case_id") or f"row_{index}"),
        "question": str(record.get("question") or ""),
        "contexts": [
            str(context)[:1600]
            for context in list(record.get("contexts") or [])[:5]
        ],
        "answer": str(record.get("answer") or ""),
    }


def _prompt(payload: list[dict[str, Any]]) -> str:
    return (
        "你是RAG回答行为裁判。不要读取或猜测数据集标签，只根据问题、"
        "检索上下文和最终回答分类。\n"
        "supported_answer：回答给出了实质内容，且所有关键事实都能由上下文支持。\n"
        "abstention：回答明确说明现有资料不足，且没有继续给出猜测性具体结论。\n"
        "unsupported_answer：回答给出了上下文无法支持的关键事实、数字或推断；"
        "即使先说资料不足再猜测，也属于此类。\n"
        "必须逐条返回且不得遗漏 case_id。输入：\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _invoke(structured_llm: Any, payload: list[dict[str, Any]]) -> dict[str, AnswerBehaviorVerdict]:
    response = structured_llm.invoke(_prompt(payload))
    return {item.case_id: item for item in response.items if item.case_id}


def judge_answerability_sync(
    records: list[dict],
    llm: Any,
    *,
    batch_size: int = 6,
) -> list[dict]:
    """判断最终回答是有证据回答、拒答还是无依据回答。"""
    updated = [dict(record) for record in records]
    pending_indices: list[int] = []
    for index, record in enumerate(updated):
        fallback = fallback_answer_behavior(str(record.get("answer") or ""))
        if fallback == "empty_response":
            record.update(evaluate_answer_behavior(record, fallback))
            record["answerability_judge_rationale"] = "模型未生成回答"
        else:
            pending_indices.append(index)

    if not pending_indices:
        return updated

    structured_llm = _structured_llm(llm)
    size = max(1, int(batch_size))
    for start in range(0, len(pending_indices), size):
        indices = pending_indices[start : start + size]
        payload = [_payload(updated[index], index) for index in indices]
        try:
            verdicts = _invoke(structured_llm, payload)
        except Exception as exc:
            logger.warning("回答行为裁判批次失败：%s", exc)
            verdicts = {}

        missing_indices = [
            index
            for index in indices
            if str(updated[index].get("case_id") or f"row_{index}") not in verdicts
        ]
        recovered = 0
        for index in missing_indices:
            case_id = str(updated[index].get("case_id") or f"row_{index}")
            try:
                verdict = _invoke(structured_llm, [_payload(updated[index], index)]).get(
                    case_id
                )
                if verdict is not None:
                    verdicts[case_id] = verdict
                    recovered += 1
                else:
                    logger.warning("回答行为裁判单条重试仍遗漏 case_id=%s", case_id)
            except Exception as exc:
                logger.warning(
                    "回答行为裁判单条重试失败 case_id=%s: %s", case_id, exc
                )
        if missing_indices:
            logger.info(
                "  回答行为裁判单条重试：%s/%s 条恢复",
                recovered,
                len(missing_indices),
            )

        for index in indices:
            record = updated[index]
            case_id = str(record.get("case_id") or f"row_{index}")
            verdict = verdicts.get(case_id)
            if verdict is not None:
                behavior = verdict.behavior
                rationale = verdict.rationale.strip()
            else:
                behavior = fallback_answer_behavior(str(record.get("answer") or ""))
                rationale = (
                    "结构化裁判未返回该记录；使用确定性拒答回退规则"
                    if behavior == "abstention"
                    else "结构化裁判未返回该记录"
                )
            record.update(evaluate_answer_behavior(record, behavior))
            record["answerability_judge_rationale"] = rationale

        logger.info(
            "  回答行为裁判进度：%s/%s",
            min(start + size, len(pending_indices)),
            len(pending_indices),
        )
    return updated


# 兼容旧测试和调用方。
_judge_answerability_sync = judge_answerability_sync
