"""评测运行配置和输入数据加载。"""

import json
import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_NOISE_KEYWORDS = (
    "免责声明", "版权所有", "联系我们", "客服电话", "扫码关注",
    "转载请注明", "本报告仅供", "投资者须知", "风险提示",
    "请联系", "官方网站", "邮箱", "传真",
)


@dataclass(frozen=True)
class EvalRunConfig:
    dataset: str
    output_dir: str | None
    sample_size: int
    top_n: int
    retrieval_only: bool
    deterministic_only: bool
    allow_query_cache: bool
    retrieval_mode: str
    concurrency: int
    seed: int
    model_ref: str
    debug_retrieval: bool
    skip_answerability_judge: bool


def load_dataset(path: str) -> list[dict]:
    """从 JSONL 加载问答对并过滤明显噪声问题。"""
    records: list[dict] = []
    noise_count = 0
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line.strip()) if line.strip() else None
            except json.JSONDecodeError:
                record = None
            if not record:
                continue
            question = str(record.get("question") or "").strip()
            if len(question) < 5 or any(word in question for word in _NOISE_KEYWORDS):
                noise_count += 1
                continue
            records.append(record)
    logger.info("加载完成：有效 %s 条，过滤噪声 %s 条", len(records), noise_count)
    return records


def sample_dataset(records: list[dict], size: int, seed: int = 42) -> list[dict]:
    """使用局部随机数生成器稳定采样，不修改进程全局随机状态。"""
    if size >= len(records):
        logger.info("采样数 %s >= 总量 %s，使用全部", size, len(records))
        return records
    sampled = random.Random(seed).sample(records, size)
    logger.info("已采样 %s / %s 条", len(sampled), len(records))
    return sampled
