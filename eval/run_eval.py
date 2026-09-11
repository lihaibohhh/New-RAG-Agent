"""
RAG 评测脚本 — src/eval/run_eval.py

用法:
    # 端到端评测（检索 + 生成 + RAGAS 三项指标）
    python -m eval.run_eval --dataset eval/dataset/eval_dataset.jsonl --n 50

    # 仅检索评测（跳过 LLM 生成，更快更省钱，只算 Precision + Recall）
    python -m eval.run_eval --dataset eval/dataset/eval_dataset.jsonl --n 50 --retrieval_only

    # 确定性检索评测（不调用生成/裁判 LLM，默认绕过语义查询缓存）
    python -m eval.run_eval --dataset eval/dataset/eval_dataset.jsonl --deterministic_only

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
from typing import Any, ClassVar
from pydantic import PrivateAttr
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult


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
from react_agent.rag.query import RetrievalService  # noqa: E402
from react_agent.rag.runtime import create_configured_rag_runtime  # noqa: E402
from eval.retrieval_metrics import (  # noqa: E402
    RETRIEVAL_METRIC_KEYS,
    aggregate_retrieval_metrics,
    evaluate_retrieval,
)

# load_chat_model 签名：load_chat_model(model_ref: str) -> BaseChatModel
# model_ref 格式："{provider}/{model_name}"，如 "deepseek/deepseek-chat"
from react_agent.utils.llm import load_chat_model  # noqa: E402

# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------
try:
    from datasets import Dataset as HFDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics.collections import (
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    _RAGAS_OK = True
except ImportError:
    _RAGAS_OK = False
    logger.warning("ragas 或 datasets 未安装，将跳过 RAGAS 打分。pip install ragas datasets")

# ---------------------------------------------------------------------------
# DeepSeek Markdown 剥离包装器
# ---------------------------------------------------------------------------


class _MarkdownStrippingLLM(BaseChatModel):
    """
    继承 BaseChatModel，在最底层的 _generate / _agenerate 剥离 Markdown 围栏。
    """

    # ClassVar：整个类只编译一次，不随实例重复创建
    _FENCE_RE: ClassVar[re.Pattern] = re.compile(
        r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL
    )

    _llm: Any = PrivateAttr()

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._llm = llm

    def _strip(self, content: str) -> str:
        content = content.strip()

        m = self._FENCE_RE.match(content)
        if m:
            return m.group(1).strip()

        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass

        m2 = re.search(r'\{.*\}', content, re.DOTALL)
        if m2:
            candidate = m2.group(0).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                logger.warning(f"[MarkdownStrip] 提取的片段不是合法 JSON: {candidate[:120]}")

        logger.warning(f"[MarkdownStrip] 无法提取有效 JSON，原样透传: {content[:80]}")
        return content

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        result = self._llm._generate(messages, stop=stop, **kwargs)
        for gen in result.generations:
            if isinstance(gen.message.content, str):
                gen.message.content = self._strip(gen.message.content)
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        result = await self._llm._agenerate(messages, stop=stop, **kwargs)
        for gen in result.generations:
            if isinstance(gen.message.content, str):
                gen.message.content = self._strip(gen.message.content)
        return result

    @property
    def _llm_type(self) -> str:
        return "markdown_stripping_wrapper"


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
    retrieval_service: RetrievalService,
    use_query_cache: bool = False,
    retrieval_mode: str = "hybrid",
    debug: bool = False,
) -> dict:
    """
    通过线上 RetrievalService 执行完整检索管道。

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
            logger.info(
                f"[DEBUG] 问题: {question[:40]}...\n"
                f"  stage={result.stage}, cache_hit={result.cache_hit}, "
                f"candidates={result.candidates_count}, chunks={len(result.chunks)}，"
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
        return {
            **record,
            "contexts": contexts,
            "sources": sources,
            "retrieved_items": retrieved_items,
            "retrieve_ok": True,
            "retrieval_stage": result.stage,
            "retrieval_cache_hit": result.cache_hit,
            "retrieval_timings": dict(result.timings),
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
    retrieval_service: RetrievalService,
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


# ---------------------------------------------------------------------------
# RAGAS 打分（sync，在 async main 中通过 asyncio.to_thread 调用）
# ---------------------------------------------------------------------------

def _run_ragas_sync(records: list[dict], llm: Any, retrieval_only: bool) -> list[dict]:
    """
    调用 RAGAS 计算指标，结果写回每条 record。
    retrieval_only=True 时只算 context_precision + context_recall。
    ragas_evaluate 是同步阻塞调用，通过 asyncio.to_thread 从 async main 中调用。

    【v2 修复：RAGAS 0.4.x 字段名破坏性变更】
    RAGAS 0.2+ 将所有数据集字段名重命名，传旧名字不报错但静默返回接近 0 的分数：
      question     -> user_input
      contexts     -> retrieved_contexts
      answer       -> response
      ground_truth -> reference
    """
    if not _RAGAS_OK:
        logger.warning("RAGAS 不可用，跳过打分，所有指标填 None")
        return [
            {**r, "context_precision": None, "context_recall": None, "faithfulness": None}
            for r in records
        ]

    # 【v2 修复】使用 RAGAS 0.2+ 的新字段名
    data: dict[str, list] = {
        "user_input":          [r["question"] for r in records],
        "retrieved_contexts":  [r.get("contexts", []) for r in records],
        "reference":           [str(r.get("ground_truth") or r.get("answer_ref", "")) for r in records],
    }
    if not retrieval_only:
        data["response"] = [r.get("answer", "") for r in records]

    # 统计 contexts 为空的比例，方便排查检索问题
    empty_ctx = sum(1 for c in data["retrieved_contexts"] if not c)
    if empty_ctx > 0:
        logger.warning(
            f"⚠️  {empty_ctx}/{len(records)} 条记录的 retrieved_contexts 为空，"
            f"这些条目的 Precision/Recall 将计为 0。"
            f"建议先用 --debug_retrieval 排查检索问题。"
        )

    hf_dataset = HFDataset.from_dict(data)

    # 【v3 修复】DeepSeek 等模型会在 JSON 响应外包裹 ```json ... ``` 代码块，
    # 导致 RAGAS Pydantic 解析器报 ValidationError（Invalid JSON at column 1）。
    # 先用 _MarkdownStrippingLLM 剥离代码围栏，再传入 LangchainLLMWrapper。
    stripped_llm = _MarkdownStrippingLLM(llm)
    wrapped_llm = LangchainLLMWrapper(stripped_llm)
    metrics = [
        ContextPrecision(llm=wrapped_llm),
        ContextRecall(llm=wrapped_llm),
    ]
    if not retrieval_only:
        metrics.append(Faithfulness(llm=wrapped_llm))

    logger.info(f"RAGAS 打分中（{len(records)} 条，指标：{[m.name for m in metrics]}）...")
    result = ragas_evaluate(hf_dataset, metrics=metrics)
    df = result.to_pandas()

    enriched = []
    nan_precision_count = 0
    for i, rec in enumerate(records):
        row = df.iloc[i]

        # 【v4 修复】ContextPrecision NaN 的语义：所有 contexts 均被 LLM 判为 irrelevant
        # RAGAS 的 AP@K 公式此时分母为 0，返回 NaN。
        # 语义上这等价于 precision=0（没有任何有用的 context），而非"无法计算"，
        # 因此记为 0.0 而非 None，使其参与均值统计，避免高估整体 Precision。
        cp_raw = _safe_float(row.get("context_precision"))
        if cp_raw is None:
            cp = 0.0
            nan_precision_count += 1
        else:
            cp = cp_raw

        enriched.append({
            **rec,
            "context_precision": cp,
            "context_recall":    _safe_float(row.get("context_recall")),
            "faithfulness":      _safe_float(row.get("faithfulness")) if not retrieval_only else None,
        })

    if nan_precision_count > 0:
        logger.warning(
            f"⚠️  {nan_precision_count}/{len(records)} 条 ContextPrecision 原始值为 NaN "
            f"（所有 contexts 均被判为 irrelevant，AP@K 分母=0），已记为 0.0。"
            f"若占比过高（>30%），建议提高 Reranker 阈值或减小 --top_n。"
        )
    return enriched


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
            float(record["retrieval_timings"]["total"]) * 1000
            for record in subset
            if isinstance(record.get("retrieval_timings"), dict)
            and record["retrieval_timings"].get("total") is not None
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
            "n":                 len(subset),
            **aggregate_retrieval_metrics(subset),
            **performance_of(subset),
        }

    summary: dict[str, Any] = {
        "generated_at":  datetime.now().isoformat(timespec="seconds"),
        "total_samples": len(records),
        "global":        metrics_of(records),
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
        "case_id", "question", "category", "industry", "page",
        "chunk_text_preview", "retrieval_label_mode", "gold_label_count",
        "retrieved_count", "retrieved_chunk_ids", "retrieval_stage",
        "retrieval_cache_hit", "retrieval_timings", *RETRIEVAL_METRIC_KEYS,
        "context_precision", "context_recall", "faithfulness",
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
                "retrieval_timings": json.dumps(
                    r.get("retrieval_timings", {}), ensure_ascii=False
                ),
                **{key: r.get(key) for key in RETRIEVAL_METRIC_KEYS},
                "context_precision":  r.get("context_precision"),
                "context_recall":     r.get("context_recall"),
                "faithfulness":       r.get("faithfulness"),
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
        ("no_answer_accuracy", "No-answer Accuracy"),
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
        "  成功率=%s  延迟 P50=%sms  P95=%sms",
        g["retrieve_ok_rate"],
        g["retrieval_latency_p50_ms"],
        g["retrieval_latency_p95_ms"],
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
        default=str(Path(__file__).resolve().parent / "dataset" / "eval_dataset.jsonl"),
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
        retrieval_service = rag_runtime.get_retrieval_service()
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

    # -- 4. RAGAS 打分（sync 函数通过 to_thread 调用）--------------------------
    if not args.deterministic_only:
        logger.info("RAGAS 打分...")
        records = await asyncio.to_thread(
            _run_ragas_sync,
            records,
            llm,
            args.retrieval_only,
        )

    # -- 5. 报告输出 -----------------------------------------------------------
    summary = compute_summary(records)
    summary["run"] = {
        "top_k": args.top_n,
        "query_cache_enabled": args.allow_query_cache,
        "deterministic_only": args.deterministic_only,
        "retrieval_only": args.retrieval_only,
        "retrieval_mode": args.retrieval_mode,
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
