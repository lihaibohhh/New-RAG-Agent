"""RAG 评测 CLI 和阶段编排入口。"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from eval.pipeline.answer_judge import _judge_answerability_sync
from eval.pipeline.generation import _generate_answers_sync
from eval.pipeline.models import EvalRunConfig, load_dataset, sample_dataset
from eval.pipeline.reporting import (
    compute_summary,
    print_console_report,
    write_csv_detail,
    write_json_summary,
)
from eval.pipeline.retrieval import batch_retrieve
from eval.pipeline import ragas_runner as _ragas
from react_agent.models import load_chat_model
from react_agent.rag.runtime import create_configured_rag_runtime

logger = logging.getLogger(__name__)


def _read_model_ref_from_config() -> str | None:
    config_path = (
        Path(__file__).resolve().parent.parent
        / "react_agent"
        / "configuration"
        / "config.yaml"
    )
    if not config_path.exists():
        return None
    try:
        import yaml

        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        for key in ("model_ref", "model"):
            value = config.get(key)
            if isinstance(value, str) and "/" in value:
                return value
    except Exception:
        return None
    return None

# 保留旧模块的测试/扩展注入点；新实现位于 pipeline.ragas_runner。
settings = _ragas.settings
AsyncOpenAI = _ragas.AsyncOpenAI
ragas_llm_factory = getattr(_ragas, "ragas_llm_factory", None)
ContextPrecision = getattr(_ragas, "ContextPrecision", None)
ContextRecall = getattr(_ragas, "ContextRecall", None)
Faithfulness = getattr(_ragas, "Faithfulness", None)
_RAGAS_OK = _ragas._RAGAS_OK
_ragas_build = _ragas.build_ragas_llm


def _sync_ragas_dependencies() -> None:
    _ragas.settings = settings
    _ragas.AsyncOpenAI = AsyncOpenAI
    _ragas.ragas_llm_factory = ragas_llm_factory
    _ragas.ContextPrecision = ContextPrecision
    _ragas.ContextRecall = ContextRecall
    _ragas.Faithfulness = Faithfulness
    _ragas._RAGAS_OK = _RAGAS_OK


def _build_ragas_llm(model_ref: str):
    _sync_ragas_dependencies()
    return _ragas_build(model_ref)


async def _run_ragas_async(records, model_ref, retrieval_only):
    _sync_ragas_dependencies()
    _ragas.build_ragas_llm = _build_ragas_llm
    return await _ragas.run_ragas(records, model_ref, retrieval_only)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 系统自动评测工具")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "dataset" / "eval_dataset_docling_v1.jsonl"),
        help=".jsonl 评测数据集路径",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("EVAL_RESULTS_DIR"),
        help="报告输出目录；默认读取 EVAL_RESULTS_DIR，否则写入 eval/results",
    )
    parser.add_argument("--n", type=int, default=150, help="随机采样条数")
    parser.add_argument("--top_n", type=int, default=3, help="最终返回文档数")
    parser.add_argument("--retrieval_only", action="store_true")
    parser.add_argument("--deterministic_only", action="store_true")
    parser.add_argument("--allow_query_cache", action="store_true")
    parser.add_argument(
        "--retrieval_mode",
        choices=("hybrid", "bm25", "vector"),
        default="hybrid",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--debug_retrieval", action="store_true")
    parser.add_argument("--skip_answerability_judge", action="store_true")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> EvalRunConfig:
    return EvalRunConfig(
        dataset=args.dataset,
        output_dir=args.output_dir,
        sample_size=args.n,
        top_n=max(1, min(args.top_n, 10)),
        retrieval_only=args.retrieval_only,
        deterministic_only=args.deterministic_only,
        allow_query_cache=args.allow_query_cache,
        retrieval_mode=args.retrieval_mode,
        concurrency=args.concurrency,
        seed=args.seed,
        model_ref=args.model or _read_model_ref_from_config() or "",
        debug_retrieval=args.debug_retrieval,
        skip_answerability_judge=args.skip_answerability_judge,
    )


async def main() -> None:
    """按配置依次执行检索、生成、裁判、RAGAS 和报告阶段。"""
    config = _config_from_args(parse_args())
    llm = None
    if config.deterministic_only:
        logger.info("确定性评测模式：不加载生成或裁判 LLM")
    else:
        if not config.model_ref:
            logger.error("未找到模型配置，请通过 --model provider/model_name 指定")
            sys.exit(1)
        logger.info("使用模型：%s", config.model_ref)
        llm = load_chat_model(config.model_ref)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at = time.time()
    logger.info("加载数据集：%s", config.dataset)
    records = load_dataset(config.dataset)
    if not records:
        logger.error("数据集为空，退出")
        sys.exit(1)
    records = sample_dataset(records, config.sample_size, seed=config.seed)

    logger.info(
        "开始批量检索（top_n=%s，concurrency=%s）...",
        config.top_n,
        config.concurrency,
    )
    runtime = create_configured_rag_runtime()
    try:
        retrieval_service = (
            runtime.get_retrieval_service()
            if config.allow_query_cache
            else runtime.get_evaluation_retrieval_service()
        )
        if config.allow_query_cache:
            logger.warning("查询缓存已启用，本次不会产生分阶段评测 Trace")
        logger.info("评测前执行显式预热...")
        warmup_status = await runtime.operations.ensure_ready(120)
        if not warmup_status.get("ready"):
            raise RuntimeError(f"RAG 预热失败: {warmup_status}")
        records = await batch_retrieve(
            records,
            top_n=config.top_n,
            retrieval_service=retrieval_service,
            concurrency=config.concurrency,
            debug_first_n=3 if config.debug_retrieval else 0,
            use_query_cache=config.allow_query_cache,
            retrieval_mode=config.retrieval_mode,
        )
    finally:
        await runtime.close()
    logger.info(
        "检索完成：%s/%s 条无异常，%s/%s 条有 contexts",
        sum(bool(record.get("retrieve_ok")) for record in records),
        len(records),
        sum(bool(record.get("contexts")) for record in records),
        len(records),
    )

    if not config.retrieval_only and not config.deterministic_only:
        logger.info("生成答案（端到端模式）...")
        records = await asyncio.to_thread(_generate_answers_sync, records, llm)
        if not config.skip_answerability_judge:
            logger.info("评判回答/拒答行为...")
            records = await asyncio.to_thread(_judge_answerability_sync, records, llm)

    if not config.deterministic_only:
        logger.info("RAGAS 打分...")
        records = await _ragas.run_ragas(
            records,
            config.model_ref,
            config.retrieval_only,
        )

    summary = compute_summary(records)
    summary["run"] = {
        "top_k": config.top_n,
        "query_cache_enabled": config.allow_query_cache,
        "deterministic_only": config.deterministic_only,
        "retrieval_only": config.retrieval_only,
        "answerability_judge_enabled": (
            not config.deterministic_only
            and not config.retrieval_only
            and not config.skip_answerability_judge
        ),
        "retrieval_mode": config.retrieval_mode,
        "evaluation_trace_enabled": not config.allow_query_cache,
        "retrieval_configuration": next(
            (
                dict(record.get("retrieval_configuration") or {})
                for record in records
                if record.get("retrieval_configuration")
            ),
            {},
        ),
        "warmup_timings": dict(
            (warmup_status.get("warmup_status") or {}).get("timings") or {}
        ),
    }
    json_path = write_json_summary(summary, timestamp, config.output_dir)
    csv_path = write_csv_detail(records, timestamp, config.output_dir)
    print_console_report(summary)
    logger.info("评测完成，耗时 %.1fs", time.time() - started_at)
    logger.info("   摘要报告：%s", json_path)
    logger.info("   明细报告：%s", csv_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
