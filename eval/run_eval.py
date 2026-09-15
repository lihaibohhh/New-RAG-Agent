"""
RAG 评测脚本 — src/eval/run_eval.py

用法:
    # 端到端评测（检索 + 生成 + RAGAS 三项指标）
    python -m eval.run_eval --dataset eval/dataset/eval_dataset_docling_v1.jsonl --n 50

    # 仅检索评测（跳过 LLM 生成，更快更省钱，只算 Precision + Recall）
    python -m eval.run_eval --dataset eval/dataset/eval_dataset_docling_v1.jsonl --n 50 --retrieval_only

    # 确定性检索评测（不调用生成/裁判 LLM，默认绕过语义查询缓存）
    python -m eval.run_eval --dataset eval/dataset/eval_dataset_docling_v1.jsonl --deterministic_only

    # 指定模型（默认从 config.yaml 读取；若 config 无此键可通过此参数覆盖）
    python -m eval.run_eval --model deepseek/deepseek-chat --n 30

输出:
    eval_summary_<timestamp>.json   — 全局均值 + 分行业均值
    eval_detail_<timestamp>.csv     — 逐条得分明细
    控制台                          — 格式化表格 + 自动诊断建议

【修复记录 v1】
  - 检索统一通过 RetrievalService 执行，与线上 Agent/MCP 共用同一编排链
  - 修复 sync 函数调用 async 函数的静默失效问题
  - 修复 ThreadPoolExecutor 调度 async 函数导致的 RuntimeError
  - main() 改为 async + asyncio.run()；参数名 top_k -> top_n

【修复记录 v2】
  - 修复 RAGAS 0.4.x 字段名破坏性变更（静默返回 0 分，不报错）：
      question     -> user_input
      contexts     -> retrieved_contexts
      answer       -> response
      ground_truth -> reference
  - 修复 contexts_count=0 问题并补充空结果诊断
  - 新增 --debug_retrieval 参数：对前 3 条记录打印详细检索中间结果，用于诊断

【修复记录 v3】
  - 修复 _MarkdownStrippingLLM：改为继承 BaseChatModel，重写 _generate/_agenerate
    原 Wrapper 因 __getattr__ 透传，RAGAS 绕过所有公开方法直接调 _generate，
    导致剥离逻辑从未执行。继承方案从架构上占据最底层，无法被绕过。
  - 补充缺失的 ChatResult 导入

【修复记录 v4】
  - 修复 ContextPrecision NaN=71% 问题：
    根因：RAGAS AP@K 公式在所有 contexts 均被判为 irrelevant 时分母为 0 → NaN
    现象：contexts_count=5 时 NaN 率高达 89%，contexts_count=1-2 时为 0%
    修复1：NaN Precision 语义上等于 "无 context 有用" → 记为 0.0 而非排除
    修复2：fallback 时限制为 min(3, top_n) 条，避免低质量文档全部被判 irrelevant
    修复3：默认 --top_n 从 5 改为 3（与线上检索默认值一致，减少噪声 context）

【修复记录 v5】
  - RAGAS 0.4 Collections改用llm_factory返回的InstructorLLM。
  - DeepSeek通过原生AsyncOpenAI客户端复用项目中的API Key和Base URL。

【修复记录 v6】
  - 移除旧的 datasets.Dataset + ragas.evaluate 批处理执行链。
  - RAGAS 0.4 Collections 指标逐条调用 ascore()，并从 MetricResult.value 取分。
  - 无答案样本不进入 RAGAS；单项失败记录到 ragas_error，不中断整批。

【修复记录 v7】
  - 检索延迟汇总优先使用包含 Reranker 的 evaluation_total。
  - 回答行为裁判对批次遗漏的 case 执行一次单条重试。
  - 摘要和控制台显式报告 Answerability Judge Coverage。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# 路径修复：确保从 src/ 下任何位置运行都能找到 react_agent 包
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parent.parent   # src/
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# 项目内部导入
# ---------------------------------------------------------------------------
from react_agent.rag.runtime import create_configured_rag_runtime  # noqa: E402
from react_agent.core.config import settings  # noqa: E402
from eval.retrieval_metrics import (  # noqa: E402
    RETRIEVAL_METRIC_KEYS,
    aggregate_retrieval_metrics,
    evaluate_retrieval,
)
from eval.answerability_metrics import (  # noqa: E402
    aggregate_answerability_metrics,
    evaluate_answer_behavior,
)

# load_chat_model 签名：load_chat_model(model_ref: str) -> BaseChatModel
# model_ref 格式："{provider}/{model_name}"，如 "deepseek/deepseek-chat"
from react_agent.models import load_chat_model  # noqa: E402

# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------
try:
    from ragas.llms import llm_factory as ragas_llm_factory
    from ragas.metrics.collections import (
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    _RAGAS_OK = True
except ImportError:
    _RAGAS_OK = False
    logger.warning("ragas 未安装，将跳过 RAGAS 打分。pip install ragas")

# ---------------------------------------------------------------------------
# RAGAS 0.4 LLM factory
# ---------------------------------------------------------------------------


def _required_ragas_secret(name: str) -> str:
    value = str(getattr(settings.secrets, name, "") or "").strip()
    if not value:
        raise EnvironmentError(f"RAGAS 裁判缺少必要配置：{name}")
    return value


def _build_ragas_llm(model_ref: str) -> Any:
    """为RAGAS Collections指标创建现代InstructorLLM。"""
    provider, separator, model_name = str(model_ref or "").strip().partition("/")
    provider = provider.casefold()
    model_name = model_name.strip()
    if not separator or not provider or not model_name:
        raise ValueError("RAGAS 模型必须使用 provider/model_name 格式")

    client_kwargs = {
        "timeout": settings.llm.llm_timeout,
        "max_retries": settings.llm.llm_retries,
    }
    if provider in {"deepseek", "ds"}:
        client = AsyncOpenAI(
            api_key=_required_ragas_secret("DEEPSEEK_API_KEY"),
            base_url=_required_ragas_secret("DEEPSEEK_BASE_URL"),
            **client_kwargs,
        )
    elif provider == "openai":
        client = AsyncOpenAI(
            api_key=_required_ragas_secret("OPENAI_API_KEY"),
            **client_kwargs,
        )
    elif provider in {"local", "qwen-local", "openai-compatible"}:
        client = AsyncOpenAI(
            api_key=(settings.secrets.LOCAL_OPENAI_API_KEY or "local-key"),
            base_url=(
                settings.secrets.LOCAL_OPENAI_BASE_URL
                or "http://127.0.0.1:8000/v1"
            ),
            **client_kwargs,
        )
    else:
        raise ValueError(
            "RAGAS Collections当前只支持本项目的OpenAI兼容provider："
            "deepseek、openai、local、qwen-local、openai-compatible；"
            f"收到：{provider}"
        )

    return ragas_llm_factory(
        model_name,
        provider="openai",
        client=client,
        adapter="instructor",
    )


# ---------------------------------------------------------------------------
# 噪声过滤关键词
# ---------------------------------------------------------------------------
_NOISE_KEYWORDS = [
    "免责声明", "版权所有", "联系我们", "客服电话", "扫码关注",
    "转载请注明", "本报告仅供", "投资者须知", "风险提示",
    "请联系", "官方网站", "邮箱", "传真",
]

# ---------------------------------------------------------------------------
# 从 config.yaml 读取模型引用
# ---------------------------------------------------------------------------


def _read_model_ref_from_config() -> str | None:
    """
    尝试从 config.yaml 读取 LLM 模型引用。
    期望格式（在 config.yaml 中二选一）：

        model_ref: deepseek/deepseek-chat
        # 或：
        model: deepseek/deepseek-chat

    如果 config.yaml 不存在或未配置此键，返回 None。
    若你的 config.yaml 键名不同（如 llm.model），请修改下方 _CANDIDATE_KEYS。
    """
    _CANDIDATE_KEYS = ("model_ref", "model")
    config_path = _SRC_ROOT / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key in _CANDIDATE_KEYS:
            val = cfg.get(key)
            if isinstance(val, str) and "/" in val:
                return val
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """从 .jsonl 加载问答对，自动过滤噪声问题。"""
    records: list[dict] = []
    noise_count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = obj.get("question", "").strip()
            if not question:
                continue
            if any(kw in question for kw in _NOISE_KEYWORDS):
                noise_count += 1
                continue
            if len(question) < 5:
                noise_count += 1
                continue

            records.append(obj)

    logger.info(f"加载完成：有效 {len(records)} 条，过滤噪声 {noise_count} 条")
    return records


def sample_dataset(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """随机采样 n 条，n <= 总数时直接使用全部。"""
    if n >= len(records):
        logger.info(f"采样数 {n} >= 总量 {len(records)}，使用全部")
        return records
    random.seed(seed)
    sampled = random.sample(records, n)
    logger.info(f"已采样 {len(sampled)} / {len(records)} 条")
    return sampled


# ---------------------------------------------------------------------------
# 检索（async）
# ---------------------------------------------------------------------------

async def retrieve_for_one(
    record: dict,
    top_n: int,
    retrieval_service: Any,
    use_query_cache: bool = False,
    retrieval_mode: str = "hybrid",
    debug: bool = False,
) -> dict:
    """
    默认通过 Knowledge Service 的专用评测管道执行检索。

    评测不再自行拼装召回和精排；空结果也保持线上语义，避免评测指标
    与实际 Agent/MCP 行为产生偏差。
    """
    question = record["question"]
    try:
        result = await retrieval_service.search(
            question,
            top_k=top_n,
            use_query_cache=use_query_cache,
            retrieval_mode=retrieval_mode,
        )

        if debug:
            traced_stages = getattr(result, "stages", {})
            logger.info(
                f"[DEBUG] 问题: {question[:40]}...\n"
                f"  stage={getattr(result, 'stage', 'evaluation_trace')}, "
                f"cache_hit={getattr(result, 'cache_hit', False)}, "
                f"stages={{{', '.join(f'{key}: {len(value)}' for key, value in traced_stages.items())}}}, "
                f"chunks={len(result.chunks)}，"
                f"前3条预览: {[chunk.content[:50] for chunk in result.chunks[:3]]}"
            )

        contexts: list[str] = []
        sources: list[dict] = []
        retrieved_items: list[dict] = []
        for rank, chunk in enumerate(result.chunks, 1):
            page = chunk.source_page
            source_file = chunk.source_file
            prefix = (
                f"[来源：{Path(source_file).name} 第 {page} 页] "
                if source_file and page is not None
                else ""
            )
            contexts.append(prefix + chunk.content)
            sources.append(
                {"file": source_file, "page": page, "chunk_id": chunk.chunk_id}
            )
            retrieved_items.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "source_file": source_file,
                    "source_page": page,
                    "score": chunk.score,
                }
            )

        metrics = evaluate_retrieval(record, retrieved_items, top_k=top_n)
        trace_stages = getattr(result, "stages", {})
        retrieval_trace = {
            name: [
                {
                    "rank": item.rank,
                    "chunk_id": item.chunk_id,
                    "source_file": item.source_file,
                    "source_page": item.source_page,
                    "content_chars": item.content_chars,
                    "doc_type": item.doc_type,
                    "industry": item.industry,
                }
                for item in candidates
            ]
            for name, candidates in trace_stages.items()
        }
        return {
            **record,
            "contexts": contexts,
            "sources": sources,
            "retrieved_items": retrieved_items,
            "retrieve_ok": True,
            "retrieval_stage": getattr(result, "stage", "evaluation_trace"),
            "retrieval_cache_hit": getattr(result, "cache_hit", False),
            "retrieval_timings": dict(result.timings),
            "retrieval_trace": retrieval_trace,
            "retrieval_configuration": dict(
                getattr(result, "configuration", {})
            ),
            "retrieval_degraded_sources": list(
                getattr(result, "degraded_sources", ())
            ),
            **metrics,
        }

    except Exception as e:
        logger.warning(f"检索失败 [{question[:30]}...]: {e}", exc_info=True)
        metrics = evaluate_retrieval(record, [], top_k=top_n)
        return {
            **record,
            "contexts": [],
            "sources": [],
            "retrieved_items": [],
            "retrieve_ok": False,
            **metrics,
        }


async def batch_retrieve(
    records: list[dict],
    top_n: int,
    retrieval_service: Any,
    concurrency: int = 8,
    debug_first_n: int = 0,
    use_query_cache: bool = False,
    retrieval_mode: str = "hybrid",
) -> list[dict]:
    """
    并发检索所有问题，保持原始顺序。
    debug_first_n > 0 时对前 N 条打印详细检索中间结果。
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(records)
    done_count = 0

    async def bounded_retrieve(idx: int, record: dict) -> None:
        nonlocal done_count
        async with sem:
            results[idx] = await retrieve_for_one(
                record,
                top_n,
                retrieval_service,
                use_query_cache=use_query_cache,
                retrieval_mode=retrieval_mode,
                debug=(idx < debug_first_n),
            )
        done_count += 1
        if done_count % 10 == 0 or done_count == len(records):
            logger.info(f"  检索进度：{done_count}/{len(records)}")

    await asyncio.gather(*(bounded_retrieve(i, rec) for i, rec in enumerate(records)))
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 答案生成（sync，在 async main 中通过 asyncio.to_thread 调用）
# ---------------------------------------------------------------------------

def _generate_answers_sync(records: list[dict], llm: Any) -> list[dict]:
    """
    用 LLM 对每条问题生成答案（端到端模式）。
    llm.invoke 是同步调用，通过 asyncio.to_thread 从 async main 中调用，
    避免阻塞事件循环。
    """
    updated = []
    for i, rec in enumerate(records, 1):
        question = rec["question"]
        ctx_text = "\n\n".join(rec.get("contexts", [])) or "（无检索结果）"
        prompt = (
            f"根据以下参考资料，简洁准确地回答问题。如果资料中没有相关信息，请如实说明。\n\n"
            f"参考资料：\n{ctx_text}\n\n"
            f"问题：{question}\n\n答案："
        )
        try:
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"LLM 生成失败 [{question[:30]}...]: {e}")
            answer = ""
        updated.append({**rec, "answer": answer})
        if i % 10 == 0 or i == len(records):
            logger.info(f"  生成进度：{i}/{len(records)}")
    return updated


class _AnswerBehaviorVerdict(BaseModel):
    """裁判只识别回答行为，不读取数据集的 answerable 标签。"""

    case_id: str
    behavior: Literal[
        "supported_answer",
        "abstention",
        "unsupported_answer",
    ]
    rationale: str = Field(description="一句话说明分类依据")


class _AnswerBehaviorBatch(BaseModel):
    items: list[_AnswerBehaviorVerdict] = Field(default_factory=list)


_CLEAR_ABSTENTION_RE = re.compile(
    r"(资料|参考资料|上下文|知识库|所给信息).{0,12}"
    r"(不足|未提供|没有|无法|不能).{0,12}(确定|判断|回答|得出|查到)|"
    r"无法根据.{0,20}(确定|判断|回答|得出)|信息不足",
    re.IGNORECASE,
)
_SPECIFIC_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?\s*(?:%|万|亿|元|年|月|日|倍)")


def _fallback_answer_behavior(answer: str) -> str:
    """裁判失败时只识别非常明确、且没有具体数字的短拒答。"""
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


def _answerability_structured_llm(llm: Any):
    """创建兼容 DeepSeek 的结构化回答行为裁判。"""
    structured = llm.with_structured_output(
        _AnswerBehaviorBatch,
        method="function_calling",
    )
    api_base = str(getattr(llm, "openai_api_base", "") or "").casefold()
    model_name = str(getattr(llm, "model_name", "") or "").casefold()
    if "deepseek" in api_base or "deepseek" in model_name:
        structured = structured.bind(
            extra_body={"thinking": {"type": "disabled"}}
        )
    return structured


def _answerability_payload(record: dict, index: int) -> dict[str, Any]:
    return {
        "case_id": str(record.get("case_id") or f"row_{index}"),
        "question": str(record.get("question") or ""),
        "contexts": [
            str(context)[:1600]
            for context in list(record.get("contexts") or [])[:5]
        ],
        "answer": str(record.get("answer") or ""),
    }


def _answerability_prompt(payload: list[dict[str, Any]]) -> str:
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


def _invoke_answerability_judge(
    structured_llm: Any,
    payload: list[dict[str, Any]],
) -> dict[str, _AnswerBehaviorVerdict]:
    response = structured_llm.invoke(_answerability_prompt(payload))
    return {
        item.case_id: item
        for item in response.items
        if item.case_id
    }


def _judge_answerability_sync(
    records: list[dict],
    llm: Any,
    *,
    batch_size: int = 6,
) -> list[dict]:
    """判断最终回答是有证据回答、拒答还是无依据回答。"""
    updated = [dict(record) for record in records]
    pending_indices: list[int] = []
    for index, record in enumerate(updated):
        fallback = _fallback_answer_behavior(str(record.get("answer") or ""))
        if fallback == "empty_response":
            record.update(evaluate_answer_behavior(record, fallback))
            record["answerability_judge_rationale"] = "模型未生成回答"
        else:
            pending_indices.append(index)

    if not pending_indices:
        return updated

    structured_llm = _answerability_structured_llm(llm)
    size = max(1, int(batch_size))
    for start in range(0, len(pending_indices), size):
        indices = pending_indices[start : start + size]
        payload = [_answerability_payload(updated[index], index) for index in indices]
        try:
            verdicts = _invoke_answerability_judge(structured_llm, payload)
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
                retry_verdicts = _invoke_answerability_judge(
                    structured_llm,
                    [_answerability_payload(updated[index], index)],
                )
                verdict = retry_verdicts.get(case_id)
                if verdict is not None:
                    verdicts[case_id] = verdict
                    recovered += 1
                else:
                    logger.warning("回答行为裁判单条重试仍遗漏 case_id=%s", case_id)
            except Exception as exc:
                logger.warning(
                    "回答行为裁判单条重试失败 case_id=%s: %s",
                    case_id,
                    exc,
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
                behavior = _fallback_answer_behavior(
                    str(record.get("answer") or "")
                )
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


# ---------------------------------------------------------------------------
# RAGAS 0.4 Collections 打分
# ---------------------------------------------------------------------------


async def _run_ragas_async(
    records: list[dict],
    model_ref: str,
    retrieval_only: bool,
) -> list[dict]:
    """使用Collections指标逐条评分，并将MetricResult.value写回记录。"""
    base_records = [
        {
            **record,
            "context_precision": None,
            "context_recall": None,
            "faithfulness": None,
            "ragas_error": None,
        }
        for record in records
    ]
    answerable_indices = [
        index
        for index, record in enumerate(base_records)
        if record.get("answerable") is not False
    ]
    if not answerable_indices:
        logger.info("数据集没有可回答样本，跳过 RAGAS；拒答能力由独立裁判评测")
        return base_records

    if not _RAGAS_OK:
        logger.warning("RAGAS 不可用，跳过打分，所有指标填 None")
        return base_records

    empty_ctx = sum(
        not base_records[index].get("contexts") for index in answerable_indices
    )
    if empty_ctx > 0:
        logger.warning(
            f"⚠️  {empty_ctx}/{len(answerable_indices)} 条可回答记录的 "
            f"retrieved_contexts 为空，"
            f"这些条目的 Precision/Recall 将计为 0。"
            f"建议先用 --debug_retrieval 排查检索问题。"
        )

    evaluator_llm = _build_ragas_llm(model_ref)
    context_precision = ContextPrecision(llm=evaluator_llm)
    context_recall = ContextRecall(llm=evaluator_llm)
    faithfulness = Faithfulness(llm=evaluator_llm) if not retrieval_only else None
    metric_names = ["context_precision", "context_recall"]
    if faithfulness is not None:
        metric_names.append("faithfulness")

    logger.info(
        "RAGAS 仅评估可回答样本（%s/%s 条，指标：%s）...",
        len(answerable_indices),
        len(records),
        metric_names,
    )

    nan_precision_count = 0
    for position, record_index in enumerate(answerable_indices, 1):
        record = base_records[record_index]
        question = str(record.get("question") or "")
        contexts = [str(value) for value in (record.get("contexts") or [])]
        reference = str(
            record.get("ground_truth") or record.get("answer_ref", "")
        )
        errors: list[str] = []

        async def score_metric(name: str, awaitable: Any) -> float | None:
            try:
                result = await awaitable
                return _safe_float(getattr(result, "value", None))
            except Exception as exc:
                logger.warning(
                    "RAGAS %s失败 [%s...]: %s",
                    name,
                    question[:30],
                    exc,
                )
                errors.append(f"{name}: {type(exc).__name__}")
                return None

        cp = await score_metric(
            "context_precision",
            context_precision.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            ),
        )
        if cp is None and not errors:
            cp = 0.0
            nan_precision_count += 1

        recall = await score_metric(
            "context_recall",
            context_recall.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            ),
        )
        faithful = None
        if faithfulness is not None:
            faithful = await score_metric(
                "faithfulness",
                faithfulness.ascore(
                    user_input=question,
                    response=str(record.get("answer") or ""),
                    retrieved_contexts=contexts,
                ),
            )

        record.update(
            {
                "context_precision": cp,
                "context_recall": recall,
                "faithfulness": faithful,
                "ragas_error": "; ".join(errors) or None,
            }
        )
        if position % 5 == 0 or position == len(answerable_indices):
            logger.info("  RAGAS进度：%s/%s", position, len(answerable_indices))

    if nan_precision_count > 0:
        logger.warning(
            f"⚠️  {nan_precision_count}/{len(answerable_indices)} 条可回答样本的 "
            f"ContextPrecision 原始值为 NaN "
            f"（所有 contexts 均被判为 irrelevant，AP@K 分母=0），已记为 0.0。"
            f"若占比过高（>30%），建议提高 Reranker 阈值或减小 --top_n。"
        )
    return base_records


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return round(f, 4) if f == f else None  # NaN -> None
    except (TypeError, ValueError):
        return None


def compute_summary(records: list[dict]) -> dict:
    """计算确定性检索指标、RAGAS 指标和分行业均值。"""

    def mean(vals: list) -> float | None:
        clean = [v for v in vals if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else None

    def percentile(vals: list[float], ratio: float) -> float | None:
        if not vals:
            return None
        ordered = sorted(vals)
        index = max(0, math.ceil(len(ordered) * ratio) - 1)
        return round(ordered[index], 2)

    def performance_of(subset: list[dict]) -> dict[str, float | None]:
        latencies_ms = [
            float(
                record["retrieval_timings"].get("evaluation_total")
                if record["retrieval_timings"].get("evaluation_total") is not None
                else record["retrieval_timings"]["total"]
            ) * 1000
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
            "retrieval_latency_p50_ms": percentile(latencies_ms, 0.50),
            "retrieval_latency_p95_ms": percentile(latencies_ms, 0.95),
        }

    def metrics_of(subset: list[dict]) -> dict:
        return {
            "context_precision": mean([r.get("context_precision") for r in subset]),
            "context_recall":    mean([r.get("context_recall")    for r in subset]),
            "faithfulness":      mean([r.get("faithfulness")      for r in subset]),
            "ragas_error_count": sum(bool(r.get("ragas_error")) for r in subset),
            "n":                 len(subset),
            **aggregate_retrieval_metrics(subset),
            **aggregate_answerability_metrics(subset),
            **performance_of(subset),
        }

    summary: dict[str, Any] = {
        "generated_at":  datetime.now().isoformat(timespec="seconds"),
        "total_samples": len(records),
        "global":        metrics_of(records),
        "by_answerability": {
            "answerable": metrics_of(
                [record for record in records if record.get("answerable") is not False]
            ),
            "no_answer": metrics_of(
                [record for record in records if record.get("answerable") is False]
            ),
        },
        "by_industry":   {},
    }

    industry_groups: dict[str, list] = defaultdict(list)
    for r in records:
        ind = r.get("industry", "").strip()
        if ind:
            industry_groups[ind].append(r)

    for ind, group in industry_groups.items():
        if len(group) < 3:
            logger.debug(f"行业 [{ind}] 样本 {len(group)} 条，不足 3 条跳过")
            continue
        summary["by_industry"][ind] = metrics_of(group)

    return summary


def _output_directory(configured: str | Path | None = None) -> Path:
    value = configured or os.getenv("EVAL_RESULTS_DIR")
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parent / "results"


def write_json_summary(
    summary: dict,
    timestamp: str,
    output_dir: str | Path | None = None,
) -> Path:
    out_dir = _output_directory(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval_summary_{timestamp}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_csv_detail(
    records: list[dict],
    timestamp: str,
    output_dir: str | Path | None = None,
) -> Path:
    out_dir = _output_directory(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval_detail_{timestamp}.csv"
    fieldnames = [
        "case_id", "answerable", "question", "category", "industry", "page",
        "chunk_text_preview", "retrieval_label_mode", "gold_label_count",
        "retrieved_count", "retrieved_chunk_ids", "retrieval_stage",
        "retrieval_cache_hit", "retrieval_abstained", "retrieval_timings",
        "retrieval_trace", "retrieval_configuration",
        "retrieval_degraded_sources",
        *RETRIEVAL_METRIC_KEYS,
        "answer", "answer_behavior", "answerability_outcome",
        "answerability_correct", "correct_abstention", "hallucination",
        "false_refusal", "negative_label_conflict",
        "answerability_judge_rationale",
        "context_precision", "context_recall", "faithfulness", "ragas_error",
        "retrieve_ok", "contexts_count", "sources",
    ]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            ctxs = r.get("contexts", [])
            chunk_preview = r.get("chunk_text", "")[:80].replace("\n", " ")
            writer.writerow({
                "case_id":           r.get("case_id", ""),
                "answerable":        r.get("answerable", True),
                "question":           r.get("question", ""),
                "category":           r.get("category", ""),
                "industry":           r.get("industry", ""),
                "page":               r.get("page", ""),
                "chunk_text_preview": chunk_preview,
                "retrieval_label_mode": r.get("retrieval_label_mode"),
                "gold_label_count":   r.get("gold_label_count"),
                "retrieved_count":    r.get("retrieved_count"),
                "retrieved_chunk_ids": json.dumps(
                    [item.get("chunk_id") for item in r.get("retrieved_items", [])],
                    ensure_ascii=False,
                ),
                "retrieval_stage":    r.get("retrieval_stage"),
                "retrieval_cache_hit": r.get("retrieval_cache_hit"),
                "retrieval_abstained": r.get("retrieval_abstained"),
                "retrieval_timings": json.dumps(
                    r.get("retrieval_timings", {}), ensure_ascii=False
                ),
                "retrieval_trace": json.dumps(
                    r.get("retrieval_trace", {}), ensure_ascii=False
                ),
                "retrieval_configuration": json.dumps(
                    r.get("retrieval_configuration", {}), ensure_ascii=False
                ),
                "retrieval_degraded_sources": json.dumps(
                    r.get("retrieval_degraded_sources", []), ensure_ascii=False
                ),
                **{key: r.get(key) for key in RETRIEVAL_METRIC_KEYS},
                "answer": r.get("answer"),
                "answer_behavior": r.get("answer_behavior"),
                "answerability_outcome": r.get("answerability_outcome"),
                "answerability_correct": r.get("answerability_correct"),
                "correct_abstention": r.get("correct_abstention"),
                "hallucination": r.get("hallucination"),
                "false_refusal": r.get("false_refusal"),
                "negative_label_conflict": r.get("negative_label_conflict"),
                "answerability_judge_rationale": r.get(
                    "answerability_judge_rationale"
                ),
                "context_precision":  r.get("context_precision"),
                "context_recall":     r.get("context_recall"),
                "faithfulness":       r.get("faithfulness"),
                "ragas_error":        r.get("ragas_error"),
                "retrieve_ok":        r.get("retrieve_ok"),
                "contexts_count":     len(ctxs),
                "sources":            json.dumps(r.get("sources", []), ensure_ascii=False),
            })
    return out


def print_console_report(summary: dict) -> None:
    """控制台打印格式化表格 + 自动诊断建议。"""
    g = summary["global"]
    sep = "─" * 56

    logger.info(f"\n{'=' * 56}")
    logger.info(f"  RAG 评测报告  |  {summary['generated_at']}")
    logger.info(f"{'=' * 56}")
    logger.info(f"  总样本数：{summary['total_samples']}")
    logger.info(sep)
    logger.info("  确定性检索指标")
    logger.info(sep)
    for key, label in [
        ("hit_at_k", "Hit@K"),
        ("precision_at_k", "Precision@K"),
        ("recall_at_k", "Recall@K"),
        ("mrr", "MRR"),
        ("ndcg_at_k", "nDCG@K"),
        ("source_page_recall", "Source/Page Recall"),
    ]:
        val = g[key]
        display = f"{val:.4f}" if val is not None else "  N/A "
        logger.info(f"  {label:<22} {display:>8}")
    logger.info(
        "  可确定性评测：%s，缺少标签：%s，无答案样本：%s",
        g["retrieval_evaluable_count"],
        g["retrieval_missing_label_count"],
        g["no_answer_count"],
    )
    logger.info(
        "  负样本检索抑制率=%s  检索异常=%s",
        g.get("retrieval_abstention_rate"),
        g.get("no_answer_retrieval_error_count"),
    )
    logger.info(
        "  成功率=%s  端到端延迟 P50=%sms  P95=%sms",
        g["retrieve_ok_rate"],
        g["retrieval_latency_p50_ms"],
        g["retrieval_latency_p95_ms"],
    )
    logger.info(sep)

    logger.info("  Agent 回答/拒答指标")
    logger.info(sep)
    for key, label in [
        ("answerability_accuracy", "Answerability Accuracy"),
        ("abstention_accuracy", "Abstention Accuracy"),
        ("hallucination_rate", "Hallucination Rate"),
        ("false_refusal_rate", "False Refusal Rate"),
        ("negative_label_conflict_rate", "Negative Label Conflict"),
    ]:
        val = g.get(key)
        display = f"{val:.4f}" if val is not None else "  N/A "
        logger.info(f"  {label:<26} {display:>8}")
    logger.info(
        "  可评判=%s  Judge Coverage=%s  空回答=%s  裁判失败=%s",
        g.get("answerability_evaluable_count"),
        g.get("answerability_judge_coverage"),
        g.get("empty_response_count"),
        g.get("answerability_judge_error_count"),
    )
    logger.info(sep)

    logger.info("  RAGAS 指标")
    logger.info(sep)
    logger.info(f"  {'指标':<22} {'均值':>8}")
    logger.info(sep)
    for key, label in [
        ("context_precision", "Context Precision"),
        ("context_recall",    "Context Recall"),
        ("faithfulness",      "Faithfulness"),
    ]:
        val = g[key]
        display = f"{val:.4f}" if val is not None else "  N/A "
        logger.info(f"  {label:<22} {display:>8}")
    logger.info("  RAGAS 单样本失败数：%s", g.get("ragas_error_count", 0))
    logger.info(sep)

    if summary["by_industry"]:
        logger.info("\n  分行业统计（样本 >= 3）")
        logger.info(sep)
        logger.info(f"  {'行业':<14} {'Precision':>10} {'Recall':>10} {'Faithful':>10} {'n':>4}")
        logger.info(sep)
        for ind, m in sorted(summary["by_industry"].items()):
            p = f"{m['context_precision']:.3f}" if m["context_precision"] is not None else " N/A"
            r = f"{m['context_recall']:.3f}" if m["context_recall"] is not None else " N/A"
            fa = f"{m['faithfulness']:.3f}" if m["faithfulness"] is not None else " N/A"
            logger.info(f"  {ind:<14} {p:>10} {r:>10} {fa:>10} {m['n']:>4}")
        logger.info(sep)

    logger.info("\n  自动诊断建议")
    logger.info(sep)
    prec = g["context_precision"]
    rec = g["context_recall"]
    faith = g["faithfulness"]
    has_suggestion = False

    if rec is not None and rec < 0.5:
        logger.info("  Recall 偏低 -> 建议扩大 top_n（当前可调 --top_n 参数）")
        logger.info("       或检查 chunk_size 是否过小导致关键段落碎片化")
        has_suggestion = True
    if prec is not None and prec < 0.5:
        logger.info("  Precision 偏低 -> 召回噪声多，可提高 Reranker 阈值")
        logger.info("       (RERANKER_THRESHOLD 当前建议 -5，可适当上调至 -3)")
        logger.info("       或减小 --top_n（当前默认 3），减少低质量 context 干扰")
        has_suggestion = True
    if faith is not None and faith < 0.6:
        logger.info("  Faithfulness 偏低 -> 答案幻觉风险高，建议检查 system prompt")
        logger.info("       或缩短 context 截断长度，减少低质量片段干扰")
        has_suggestion = True
    if not has_suggestion:
        logger.info("  各项指标正常，无明显异常。")

    logger.info(f"{'=' * 56}\n")


# ---------------------------------------------------------------------------
# 主入口（async）
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 系统自动评测工具")
    parser.add_argument(
        "--dataset",
        default=str(
            Path(__file__).resolve().parent
            / "dataset"
            / "eval_dataset_docling_v1.jsonl"
        ),
        help=".jsonl 评测数据集路径",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("EVAL_RESULTS_DIR"),
        help="报告输出目录；默认读取 EVAL_RESULTS_DIR，否则写入 eval/results",
    )
    parser.add_argument("--n",    type=int, default=150,  help="随机采样条数（默认 150）")
    parser.add_argument(
        "--top_n", type=int, default=3,
        help="统一检索服务返回 Top-N 文档数（默认 3，与线上默认值一致）",
    )
    parser.add_argument("--retrieval_only", action="store_true",
        help="只测检索质量，跳过 LLM 生成与 Faithfulness")
    parser.add_argument(
        "--deterministic_only",
        action="store_true",
        help="只计算 gold 标签指标，不加载生成或裁判 LLM",
    )
    parser.add_argument(
        "--allow_query_cache",
        action="store_true",
        help="允许命中线上语义查询缓存；A/B 评测默认禁用，通常不要开启",
    )
    parser.add_argument(
        "--retrieval_mode",
        choices=("hybrid", "bm25", "vector"),
        default="hybrid",
        help="选择完整混合、纯 BM25 或纯向量候选源",
    )
    parser.add_argument("--concurrency", type=int, default=8,
        help="并发检索协程数（默认 8）")
    parser.add_argument("--seed", type=int, default=42,  # 42
        help="随机种子，保证采样可复现（默认 42）")
    parser.add_argument(
        "--model", default='deepseek/deepseek-chat',
        help=(
            "LLM 模型引用，格式 provider/model_name（如 deepseek/deepseek-v4-flash）。"
            "不传时自动从 config.yaml 读取 model_ref 或 model 键。"
        ),
    )
    parser.add_argument(
        "--debug_retrieval", action="store_true",
        help="对前 3 条记录打印统一检索服务的阶段、缓存和候选数，用于排查 contexts 为空问题",
    )
    parser.add_argument(
        "--skip_answerability_judge",
        action="store_true",
        help="端到端模式下跳过回答/拒答裁判；默认启用并会增加模型调用",
    )
    args = parser.parse_args()
    args.top_n = max(1, min(args.top_n, 10))

    # -- 0. 确定模型引用 -------------------------------------------------------
    llm = None
    if not args.deterministic_only:
        model_ref = args.model or _read_model_ref_from_config()
        if not model_ref:
            logger.error(
                "未找到模型配置。请在 config.yaml 中设置 model_ref 键，"
                "或通过 --model provider/model_name 参数传入。"
            )
            sys.exit(1)
        logger.info(f"使用模型：{model_ref}")
        llm = load_chat_model(model_ref)   # @lru_cache，多次调用不会重复初始化
    else:
        logger.info("确定性评测模式：不加载生成或裁判 LLM")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start = time.time()

    # -- 1. 加载数据 -----------------------------------------------------------
    logger.info(f"加载数据集：{args.dataset}")
    records = load_dataset(args.dataset)
    if not records:
        logger.error("数据集为空，退出")
        sys.exit(1)
    records = sample_dataset(records, args.n, seed=args.seed)

    # -- 2. 批量检索（async，Semaphore 控并发）---------------------------------
    logger.info(f"开始批量检索（top_n={args.top_n}，concurrency={args.concurrency}）...")
    debug_n = 3 if args.debug_retrieval else 0
    rag_runtime = create_configured_rag_runtime()
    try:
        retrieval_service = (
            rag_runtime.get_retrieval_service()
            if args.allow_query_cache
            else rag_runtime.get_evaluation_retrieval_service()
        )
        if args.allow_query_cache:
            logger.warning(
                "--allow_query_cache 使用普通检索端点，"
                "本次不会产生分阶段评测 trace"
            )
        logger.info("评测前执行显式预热...")
        warmup_status = await rag_runtime.operations.ensure_ready(120)
        if not warmup_status.get("ready"):
            raise RuntimeError(f"RAG 预热失败: {warmup_status}")
        records = await batch_retrieve(
            records,
            top_n=args.top_n,
            retrieval_service=retrieval_service,
            concurrency=args.concurrency,
            debug_first_n=debug_n,
            use_query_cache=args.allow_query_cache,
            retrieval_mode=args.retrieval_mode,
        )
    finally:
        await rag_runtime.close()
    retrieved_ok = sum(1 for r in records if r.get("retrieve_ok"))
    contexts_ok = sum(1 for r in records if r.get("contexts"))
    logger.info(f"检索完成：{retrieved_ok}/{len(records)} 条无异常，{contexts_ok}/{len(records)} 条有 contexts")

    # -- 3. LLM 生成答案（端到端模式，sync 函数通过 to_thread 调用）-----------
    if not args.retrieval_only and not args.deterministic_only:
        logger.info("生成答案（端到端模式）...")
        records = await asyncio.to_thread(_generate_answers_sync, records, llm)
        if not args.skip_answerability_judge:
            logger.info("评判回答/拒答行为...")
            records = await asyncio.to_thread(
                _judge_answerability_sync,
                records,
                llm,
            )

    # -- 4. RAGAS 0.4 Collections打分 ----------------------------------------
    if not args.deterministic_only:
        logger.info("RAGAS 打分...")
        records = await _run_ragas_async(
            records,
            model_ref,
            args.retrieval_only,
        )

    # -- 5. 报告输出 -----------------------------------------------------------
    summary = compute_summary(records)
    summary["run"] = {
        "top_k": args.top_n,
        "query_cache_enabled": args.allow_query_cache,
        "deterministic_only": args.deterministic_only,
        "retrieval_only": args.retrieval_only,
        "answerability_judge_enabled": (
            not args.deterministic_only
            and not args.retrieval_only
            and not args.skip_answerability_judge
        ),
        "retrieval_mode": args.retrieval_mode,
        "evaluation_trace_enabled": not args.allow_query_cache,
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
    json_path = write_json_summary(summary, timestamp, args.output_dir)
    csv_path = write_csv_detail(records, timestamp, args.output_dir)
    print_console_report(summary)

    elapsed = time.time() - t_start
    logger.info(f"评测完成，耗时 {elapsed:.1f}s")
    logger.info(f"   摘要报告：{json_path}")
    logger.info(f"   明细报告：{csv_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
