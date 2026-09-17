"""Agent 回答行为与数据集可回答标签之间的确定性指标。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ANSWER_BEHAVIORS = (
    "supported_answer",
    "abstention",
    "unsupported_answer",
    "empty_response",
    "judge_error",
)

ANSWERABILITY_METRIC_KEYS = (
    "answerability_accuracy",
    "abstention_accuracy",
    "hallucination_rate",
    "false_refusal_rate",
    "negative_label_conflict_rate",
)


def evaluate_answer_behavior(
    record: Mapping[str, Any],
    behavior: str,
) -> dict[str, Any]:
    """把与标签无关的回答行为映射为有监督评测结果。"""
    normalized = str(behavior or "judge_error").strip().casefold()
    if normalized not in ANSWER_BEHAVIORS:
        normalized = "judge_error"

    answerable = record.get("answerable") is not False
    # 空回答是系统可观察到的失败，应进入准确率分母；只有裁判自身失败时
    # 才无法判断回答行为并从比率中排除。
    evaluable = normalized != "judge_error"
    if normalized in {"empty_response", "judge_error"}:
        outcome = normalized
    elif answerable and normalized == "supported_answer":
        outcome = "correct_answer"
    elif answerable and normalized == "abstention":
        outcome = "false_refusal"
    elif answerable:
        outcome = "unsupported_answer"
    elif normalized == "abstention":
        outcome = "correct_abstention"
    elif normalized == "supported_answer":
        outcome = "negative_label_conflict"
    else:
        outcome = "hallucinated_answer"

    return {
        "answer_behavior": normalized,
        "answerability_evaluable": evaluable,
        "answerability_outcome": outcome,
        "answerability_correct": (
            1.0
            if outcome in {"correct_answer", "correct_abstention"}
            else (0.0 if evaluable else None)
        ),
        "correct_abstention": (
            1.0 if outcome == "correct_abstention" else (0.0 if not answerable and evaluable else None)
        ),
        "hallucination": (
            1.0 if outcome == "hallucinated_answer" else (0.0 if not answerable and evaluable else None)
        ),
        "false_refusal": (
            1.0 if outcome == "false_refusal" else (0.0 if answerable and evaluable else None)
        ),
        "negative_label_conflict": (
            1.0 if outcome == "negative_label_conflict" else (0.0 if not answerable and evaluable else None)
        ),
    }


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None


def aggregate_answerability_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """聚合端到端拒答指标，并显式报告未成功评判的记录。"""
    evaluable_count = sum(
        bool(record.get("answerability_evaluable")) for record in records
    )
    return {
        "answerability_evaluable_count": evaluable_count,
        "answerability_judge_coverage": (
            round(evaluable_count / len(records), 4) if records else None
        ),
        "answerability_positive_count": sum(
            record.get("answerable") is not False for record in records
        ),
        "answerability_negative_count": sum(
            record.get("answerable") is False for record in records
        ),
        "empty_response_count": sum(
            record.get("answer_behavior") == "empty_response" for record in records
        ),
        "answerability_judge_error_count": sum(
            record.get("answer_behavior") == "judge_error" for record in records
        ),
        "answerability_accuracy": _mean(records, "answerability_correct"),
        "abstention_accuracy": _mean(records, "correct_abstention"),
        "hallucination_rate": _mean(records, "hallucination"),
        "false_refusal_rate": _mean(records, "false_refusal"),
        "negative_label_conflict_rate": _mean(
            records,
            "negative_label_conflict",
        ),
    }


__all__ = [
    "ANSWERABILITY_METRIC_KEYS",
    "ANSWER_BEHAVIORS",
    "aggregate_answerability_metrics",
    "evaluate_answer_behavior",
]
