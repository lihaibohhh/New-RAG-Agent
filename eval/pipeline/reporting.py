"""评测结果聚合，以及 JSON/CSV 报告输出。"""

import csv
import json
import math
import os
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.pipeline.metrics.answerability import aggregate_answerability_metrics
from eval.pipeline.metrics.retrieval import (
    RETRIEVAL_METRIC_KEYS,
    aggregate_retrieval_metrics,
)

logger = logging.getLogger(__name__)


def _mean(values: list) -> float | None:
    clean = [value for value in values if value is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * ratio) - 1)
    return round(ordered[index], 2)


def _performance(subset: list[dict]) -> dict[str, float | None]:
    latencies_ms = [
        float(
            record["retrieval_timings"].get("evaluation_total")
            if record["retrieval_timings"].get("evaluation_total") is not None
            else record["retrieval_timings"]["total"]
        )
        * 1000
        for record in subset
        if isinstance(record.get("retrieval_timings"), dict)
        and (
            record["retrieval_timings"].get("evaluation_total") is not None
            or record["retrieval_timings"].get("total") is not None
        )
    ]
    return {
        "retrieve_ok_rate": (
            round(sum(bool(record.get("retrieve_ok")) for record in subset) / len(subset), 4)
            if subset
            else None
        ),
        "retrieval_latency_p50_ms": _percentile(latencies_ms, 0.50),
        "retrieval_latency_p95_ms": _percentile(latencies_ms, 0.95),
    }


def _metrics(subset: list[dict]) -> dict:
    return {
        "context_precision": _mean([r.get("context_precision") for r in subset]),
        "context_recall": _mean([r.get("context_recall") for r in subset]),
        "faithfulness": _mean([r.get("faithfulness") for r in subset]),
        "ragas_error_count": sum(bool(r.get("ragas_error")) for r in subset),
        "n": len(subset),
        **aggregate_retrieval_metrics(subset),
        **aggregate_answerability_metrics(subset),
        **_performance(subset),
    }


def compute_summary(records: list[dict]) -> dict:
    """聚合确定性检索、回答行为、RAGAS 和性能指标。"""
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_samples": len(records),
        "global": _metrics(records),
        "by_answerability": {
            "answerable": _metrics(
                [record for record in records if record.get("answerable") is not False]
            ),
            "no_answer": _metrics(
                [record for record in records if record.get("answerable") is False]
            ),
        },
        "by_industry": {},
    }
    groups: dict[str, list] = defaultdict(list)
    for record in records:
        industry = str(record.get("industry") or "").strip()
        if industry:
            groups[industry].append(record)
    summary["by_industry"] = {
        industry: _metrics(group)
        for industry, group in groups.items()
        if len(group) >= 3
    }
    return summary


def _output_directory(configured: str | Path | None = None) -> Path:
    value = configured or os.getenv("EVAL_RESULTS_DIR")
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "results"


def write_json_summary(
    summary: dict,
    timestamp: str,
    output_dir: str | Path | None = None,
) -> Path:
    out_dir = _output_directory(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"eval_summary_{timestamp}.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def write_csv_detail(
    records: list[dict],
    timestamp: str,
    output_dir: str | Path | None = None,
) -> Path:
    out_dir = _output_directory(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"eval_detail_{timestamp}.csv"
    fieldnames = [
        "case_id", "answerable", "question", "category", "industry", "page",
        "chunk_text_preview", "retrieval_label_mode", "gold_label_count",
        "retrieved_count", "retrieved_chunk_ids", "retrieval_stage",
        "retrieval_cache_hit", "retrieval_abstained", "retrieval_timings",
        "retrieval_trace", "retrieval_configuration", "retrieval_degraded_sources",
        *RETRIEVAL_METRIC_KEYS,
        "answer", "answer_behavior", "answerability_outcome",
        "answerability_correct", "correct_abstention", "hallucination",
        "false_refusal", "negative_label_conflict", "answerability_judge_rationale",
        "context_precision", "context_recall", "faithfulness", "ragas_error",
        "retrieve_ok", "contexts_count", "sources",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row.update(
                {
                    "chunk_text_preview": str(record.get("chunk_text") or "")[:80].replace("\n", " "),
                    "retrieved_chunk_ids": json.dumps(
                        [item.get("chunk_id") for item in record.get("retrieved_items", [])],
                        ensure_ascii=False,
                    ),
                    "retrieval_timings": json.dumps(record.get("retrieval_timings", {}), ensure_ascii=False),
                    "retrieval_trace": json.dumps(record.get("retrieval_trace", {}), ensure_ascii=False),
                    "retrieval_configuration": json.dumps(record.get("retrieval_configuration", {}), ensure_ascii=False),
                    "retrieval_degraded_sources": json.dumps(record.get("retrieval_degraded_sources", []), ensure_ascii=False),
                    "contexts_count": len(record.get("contexts", [])),
                    "sources": json.dumps(record.get("sources", []), ensure_ascii=False),
                }
            )
            writer.writerow(row)
    return output


def print_console_report(summary: dict) -> None:
    """在控制台打印完整分区报告和自动诊断建议。"""
    metrics = summary["global"]
    separator = "─" * 56

    logger.info("\n%s", "=" * 56)
    logger.info("  RAG 评测报告  |  %s", summary["generated_at"])
    logger.info("%s", "=" * 56)
    logger.info("  总样本数：%s", summary["total_samples"])
    logger.info(separator)

    logger.info("  确定性检索指标")
    logger.info(separator)
    for key, label in (
        ("hit_at_k", "Hit@K"),
        ("precision_at_k", "Precision@K"),
        ("recall_at_k", "Recall@K"),
        ("mrr", "MRR"),
        ("ndcg_at_k", "nDCG@K"),
        ("source_page_recall", "Source/Page Recall"),
    ):
        value = metrics.get(key)
        display = f"{value:.4f}" if value is not None else "N/A"
        logger.info("  %-22s %8s", label, display)
    logger.info(
        "  可确定性评测：%s，缺少标签：%s，无答案样本：%s",
        metrics.get("retrieval_evaluable_count"),
        metrics.get("retrieval_missing_label_count"),
        metrics.get("no_answer_count"),
    )
    logger.info(
        "  负样本检索抑制率=%s  检索异常=%s",
        metrics.get("retrieval_abstention_rate"),
        metrics.get("no_answer_retrieval_error_count"),
    )
    logger.info(
        "  成功率=%s  端到端延迟 P50=%sms  P95=%sms",
        metrics.get("retrieve_ok_rate"),
        metrics.get("retrieval_latency_p50_ms"),
        metrics.get("retrieval_latency_p95_ms"),
    )
    logger.info(separator)

    logger.info("  Agent 回答/拒答指标")
    logger.info(separator)
    for key, label in (
        ("answerability_accuracy", "Answerability Accuracy"),
        ("abstention_accuracy", "Abstention Accuracy"),
        ("hallucination_rate", "Hallucination Rate"),
        ("false_refusal_rate", "False Refusal Rate"),
        ("negative_label_conflict_rate", "Negative Label Conflict"),
    ):
        value = metrics.get(key)
        display = f"{value:.4f}" if value is not None else "N/A"
        logger.info("  %-26s %8s", label, display)
    logger.info(
        "  可评判=%s  Judge Coverage=%s  空回答=%s  裁判失败=%s",
        metrics.get("answerability_evaluable_count"),
        metrics.get("answerability_judge_coverage"),
        metrics.get("empty_response_count"),
        metrics.get("answerability_judge_error_count"),
    )
    logger.info(separator)

    logger.info("  RAGAS 指标")
    logger.info(separator)
    logger.info("  %-22s %8s", "指标", "均值")
    logger.info(separator)
    for key, label in (
        ("context_precision", "Context Precision"),
        ("context_recall", "Context Recall"),
        ("faithfulness", "Faithfulness"),
    ):
        value = metrics.get(key)
        display = f"{value:.4f}" if value is not None else "N/A"
        logger.info("  %-22s %8s", label, display)
    logger.info("  RAGAS 单样本失败数：%s", metrics.get("ragas_error_count", 0))
    logger.info(separator)

    if summary.get("by_industry"):
        logger.info("\n  分行业统计（样本 >= 3）")
        logger.info(separator)
        logger.info(
            "  %-14s %10s %10s %10s %4s",
            "行业",
            "Precision",
            "Recall",
            "Faithful",
            "n",
        )
        logger.info(separator)
        for industry, industry_metrics in sorted(summary["by_industry"].items()):
            precision = industry_metrics.get("context_precision")
            recall = industry_metrics.get("context_recall")
            faithful = industry_metrics.get("faithfulness")
            logger.info(
                "  %-14s %10s %10s %10s %4s",
                industry,
                f"{precision:.3f}" if precision is not None else "N/A",
                f"{recall:.3f}" if recall is not None else "N/A",
                f"{faithful:.3f}" if faithful is not None else "N/A",
                industry_metrics.get("n", 0),
            )
        logger.info(separator)

    logger.info("\n  自动诊断建议")
    logger.info(separator)
    precision = metrics.get("context_precision")
    recall = metrics.get("context_recall")
    faithful = metrics.get("faithfulness")
    has_suggestion = False

    if recall is not None and recall < 0.5:
        logger.info("  Recall 偏低 -> 建议扩大 top_n，或检查 Chunk 是否过度碎片化")
        has_suggestion = True
    if precision is not None and precision < 0.5:
        logger.info("  Precision 偏低 -> 可提高 Reranker 阈值或减小 top_n")
        has_suggestion = True
    if faithful is not None and faithful < 0.6:
        logger.info("  Faithfulness 偏低 -> 建议检查回答 Prompt 和低质量 Context")
        has_suggestion = True
    if metrics.get("retrieve_ok_rate") not in (None, 1.0):
        logger.info("  存在检索异常 -> 建议从明细 CSV 的 retrieve_ok 和 Trace 定位")
        has_suggestion = True
    if metrics.get("answerability_judge_coverage") not in (None, 1.0):
        logger.info("  裁判覆盖率不足 -> 建议检查裁判失败记录及单条重试结果")
        has_suggestion = True
    if metrics.get("ragas_error_count", 0):
        logger.info("  存在 RAGAS 单项失败 -> 建议检查明细 CSV 的 ragas_error")
        has_suggestion = True
    if not has_suggestion:
        logger.info("  各项指标正常，无明显异常。")

    logger.info("%s\n", "=" * 56)
