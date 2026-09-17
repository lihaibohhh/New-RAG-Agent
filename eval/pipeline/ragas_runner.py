"""RAGAS 0.4 模型适配和指标执行。"""

import logging
from typing import Any

from openai import AsyncOpenAI

from react_agent.configuration.settings import settings

logger = logging.getLogger(__name__)

try:
    from ragas.llms import llm_factory as ragas_llm_factory
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

    _RAGAS_OK = True
except ImportError:
    _RAGAS_OK = False
    logger.warning("ragas 未安装，将跳过 RAGAS 打分。pip install ragas")


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
        return round(number, 4) if number == number else None
    except (TypeError, ValueError):
        return None


def _required_secret(name: str) -> str:
    value = str(getattr(settings.secrets, name, "") or "").strip()
    if not value:
        raise EnvironmentError(f"RAGAS 裁判缺少必要配置：{name}")
    return value


def build_ragas_llm(model_ref: str) -> Any:
    """为 RAGAS Collections 指标创建 InstructorLLM。"""
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
            api_key=_required_secret("DEEPSEEK_API_KEY"),
            base_url=_required_secret("DEEPSEEK_BASE_URL"),
            **client_kwargs,
        )
    elif provider == "openai":
        client = AsyncOpenAI(
            api_key=_required_secret("OPENAI_API_KEY"),
            **client_kwargs,
        )
    elif provider in {"local", "qwen-local", "openai-compatible"}:
        client = AsyncOpenAI(
            api_key=settings.secrets.LOCAL_OPENAI_API_KEY or "local-key",
            base_url=(
                settings.secrets.LOCAL_OPENAI_BASE_URL
                or "http://127.0.0.1:8000/v1"
            ),
            **client_kwargs,
        )
    else:
        raise ValueError(
            "RAGAS Collections 当前只支持 deepseek、openai、local、"
            f"qwen-local、openai-compatible；收到：{provider}"
        )

    return ragas_llm_factory(
        model_name,
        provider="openai",
        client=client,
        adapter="instructor",
    )


async def run_ragas(
    records: list[dict],
    model_ref: str,
    retrieval_only: bool,
) -> list[dict]:
    """逐条计算 RAGAS Collections 指标并写回记录。"""
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

    empty_contexts = sum(
        not base_records[index].get("contexts") for index in answerable_indices
    )
    if empty_contexts:
        logger.warning(
            "%s/%s 条可回答记录的 retrieved_contexts 为空，这些条目将计为 0",
            empty_contexts,
            len(answerable_indices),
        )

    evaluator_llm = build_ragas_llm(model_ref)
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
        contexts = [str(value) for value in record.get("contexts") or []]
        reference = str(record.get("ground_truth") or record.get("answer_ref", ""))
        errors: list[str] = []

        async def score_metric(name: str, awaitable: Any) -> float | None:
            try:
                result = await awaitable
                return _safe_float(getattr(result, "value", None))
            except Exception as exc:
                logger.warning("RAGAS %s失败 [%s...]: %s", name, question[:30], exc)
                errors.append(f"{name}: {type(exc).__name__}")
                return None

        precision = await score_metric(
            "context_precision",
            context_precision.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            ),
        )
        if precision is None and not errors:
            precision = 0.0
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
                "context_precision": precision,
                "context_recall": recall,
                "faithfulness": faithful,
                "ragas_error": "; ".join(errors) or None,
            }
        )
        if position % 5 == 0 or position == len(answerable_indices):
            logger.info("  RAGAS进度：%s/%s", position, len(answerable_indices))

    if nan_precision_count:
        logger.warning(
            "%s/%s 条 ContextPrecision 为 NaN，已按无相关 Context 记为 0.0",
            nan_precision_count,
            len(answerable_indices),
        )
    return base_records


_build_ragas_llm = build_ragas_llm
_run_ragas_async = run_ragas
