"""
dataset_generator.py
────────────────────
从已有 PDF 研报 chunks 中，用 LLM 自动生成评测 QA 对。

输出格式（JSONL，每行一条）：
{
  "schema_version": 3,
  "case_id":      "rag_a1b2c3d4e5f6",
  "question":     "比亚迪2023年第三季度毛利率是多少？",
  "ground_truth": "根据研报，比亚迪2023年Q3毛利率为22.1%，同比提升3.2个百分点。",
  "answerable":   true,
  "gold_chunk_ids": ["chunk-id"],
  "gold_sources": [{"source_file": "半导体/中芯国际_2023Q3点评.pdf", "pages": [5]}],
  "source_file":  "半导体/中芯国际_2023Q3点评.pdf",
  "page":         5,
  "industry":     "半导体",
  "reasoning_scope": "single_chunk",
  "gold_evidence": [{"chunk_id": "chunk-id", "quote": "原文引文"}],
  "evidence_validation": {"status": "passed"},
  "review_status": "pending",
  "chunk_text":   "...(原始chunk，评测时用于校验)..."
}

用法：
    # 在 src/ 目录下执行（cd src）
    python eval/dataset_generator.py \
        --data_dir ./data \
        --output   eval/results/eval_dataset_docling_v1.candidate.jsonl \
        --n_per_chunk 1 \
        --max_chunks 200 \
        --multi_chunk_ratio 0.25 \
        --no_answer_count 20 \
        --seed 42
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Literal

# ── 把 src/ 加入 path，复用项目已有模块 ──────────────────────────────────────
_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(_SRC.parent / ".env", override=False)

from langchain_core.documents import Document  # noqa: E402

# ★ 新增：Pydantic 导入
from pydantic import BaseModel, Field  # noqa: E402

from react_agent.models import load_chat_model  # noqa: E402
from react_agent.core.config import settings  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# ★ 核心改造 Step 1：定义数据模型
#
# 工程直觉：把你"期望 LLM 输出的形状"，用 Python 类写出来。
# 这就是你和 LLM 之间的"合同"——它必须按这个格式填，不能乱发挥。
#
# Field(description=...) 的作用：这段描述会被翻译进 JSON Schema 里，
# 模型在生成时能读到它，相当于表格里的填写说明，比在 Prompt 里叮嘱更可靠。
# ══════════════════════════════════════════════════════════════════════════════

class EvidenceQuote(BaseModel):
    """模型给出的可机械核验的原文证据。"""

    chunk_index: int = Field(description="证据所在片段编号，从 1 开始")
    quote: str = Field(description="从对应片段逐字摘录的短句，不得改写")


class QAPair(BaseModel):
    """单个问答对。"""
    question: str = Field(
        description="问题，必须包含具体的公司名称、行业或产品名，禁止使用'该公司'等模糊代词"
    )
    ground_truth: str = Field(
        description="答案，必须直接来自研报片段，禁止补充片段中未出现的数字或结论"
    )
    category: Literal[
        "numeric_lookup",
        "factual_lookup",
        "causal_reasoning",
        "comparison",
        "trend_analysis",
    ] = Field(description="问题所属的评测类别")
    evidence: list[EvidenceQuote] = Field(
        default_factory=list,
        description="支撑答案的原文摘录；多片段问题必须引用至少两个不同片段",
    )


class QAList(BaseModel):
    """LLM 每次调用的完整返回结构。

    为什么要包一层 QAList，而不直接用 list[QAPair]？
    因为 .with_structured_output() 要求顶层是一个对象，不能是裸数组。
    这是 JSON Schema 规范的约束。包一层是标准做法。
    """
    pairs: list[QAPair] = Field(
        default_factory=list,
        description="生成的QA对列表。若片段为免责声明、目录、分析师信息等无实质内容，返回空列表"
    )


class NoAnswerCase(BaseModel):
    """需要系统拒答或声明证据不足的候选问题。"""

    question: str = Field(description="自然、具体、但无法从给定语料证据回答的问题")
    category: Literal[
        "numeric_lookup",
        "factual_lookup",
        "causal_reasoning",
        "comparison",
        "trend_analysis",
    ] = Field(description="问题所属的评测类别")
    rationale: str = Field(description="为什么给定语料不足以回答，禁止虚构答案")


class NoAnswerList(BaseModel):
    cases: list[NoAnswerCase] = Field(default_factory=list)


@dataclass(frozen=True)
class EvidenceUnit:
    """一次生成调用使用的一个或多个相关 chunk。"""

    chunks: tuple[Document, ...]
    scope: Literal["single_chunk", "multi_chunk"]


# ── Prompt ────────────────────────────────────────────────────────────────────
# ★ 核心改造 Step 2：简化 Prompt
#
# 原来结尾那一大段"请严格以 JSON 数组格式输出..."被删掉了。
# 为什么可以删？因为 .with_structured_output() 在底层通过 Function Calling
# 强制了格式，模型根本没有机会输出"好的，以下是您的答案："这类废话。
# 在 Prompt 里再叮嘱一遍是多余的，而且有时反而会干扰模型的注意力。
#
# 保留的部分：8 条业务规则——这些是业务逻辑，不是格式控制，依然有价值。
# ─────────────────────────────────────────────────────────────────────────────
_QA_PROMPT = """你是一位苛刻的金融分析师，负责构建用于评估 RAG（检索增强生成）系统的高质量测试集。

请仔细阅读下面带编号的金融研报证据片段。如果证据不含实质性的商业/财务/行业分析内容（如免责声明、评级标准说明、分析师联系方式、纯文档目录），请将 pairs 字段返回空列表。

如果片段包含实质性内容，请生成 {n} 个高质量问答对，严格遵守以下规则：
1. 必须指名道姓：问题中必须明确包含具体的公司名称、行业名称或具体产品名，绝对不能使用"该公司"、"该行业"、"本项目"等模糊指代！如果片段中找不到具体实体名称，请放弃生成。
2. 拒绝元数据：不要提问关于图表编号（如"图1"）、分析师名字、报告日期等外围信息。
3. 聚焦核心业务：多提问关于营收数据、毛利率、产能规划、行业趋势、竞争格局等需要深度理解的问题。
4. 独立可答：ground_truth 必须直接且精准，不需要用户再去翻看原文档。
5. 若片段中存在表格数据，优先生成针对具体数值的问题。
6. 答案只能来自下方证据：ground_truth 的每一句话都必须能找到直接依据，严禁根据常识或训练数据补充片段中未出现的数字或结论。
7. 禁止模糊答案：ground_truth 中不允许出现"任意一个"、"例如"、"或者A或B"等不确定表述；若答案本身是列表，则必须完整列出所有列表项。
8. 问题必须多样化：当生成多个问题时，尽量同时覆盖数值查找和逻辑/机制分析，不得重复拆解同一句话。
9. category 必须从 numeric_lookup、factual_lookup、causal_reasoning、comparison、trend_analysis 中选择最符合的一项。
10. evidence 必须逐字摘录原文，并正确填写从 1 开始的 chunk_index；不要把改写后的答案当作引文。
11. {scope_instruction}

━━━ 编号证据片段 ━━━
{evidence_blocks}
━━━━━━━━━━━━━━━━━━"""

_NO_ANSWER_PROMPT = """你正在构建金融研报 RAG 系统的拒答评测集。

根据下面列出的语料主题与有限证据，生成 {n} 个自然、具体、看似合理但无法从这些证据直接回答的问题。问题可以沿用出现过的公司或行业实体，但必须询问证据中缺失的年份、指标、交易细节或因果结论。

要求：
1. 不得编造答案、数字或事实；rationale 只说明缺少哪类证据。
2. 问题必须指明实体，不能使用“该公司”“该行业”等模糊代词。
3. 各问题尽量覆盖不同实体与类别。
4. 这些只是候选无答案样本，后续还会经过全库检索筛查和人工确认。

━━━ 语料主题与有限证据 ━━━
{evidence_blocks}
━━━━━━━━━━━━━━━━━━━━"""

# ── 免责/目录类 chunk 预过滤关键词（命中任意一条则跳过 LLM 调用） ─────────────
_NOISE_PATTERNS = re.compile(
    r"免责声明|分析师声明|评级说明|风险提示.*本报告|版权所有.*禁止|执业编号|"
    r"请务必阅读正文之后的免责|本报告仅供.*客户使用|证券投资咨询业务资格",
    re.S,
)

# ──────────────── 工具函数 ──────────────────────────────────────────────────────────────────


def _get_industry(chunk: Document) -> str:
    return chunk.metadata.get("industry", "未知").strip() or "未知"


def _get_rel_src(chunk: Document, data_dir: Path) -> str:
    """计算相对于 data_dir 的路径字符串，失败时返回文件名。"""
    source_file = str(chunk.metadata.get("source_file") or "").strip()
    if source_file:
        return source_file.replace("\\", "/")
    src = chunk.metadata.get("source", "unknown")
    try:
        return str(Path(src).relative_to(data_dir)).replace("\\", "/")
    except ValueError:
        return Path(src).name


def _get_doc_type(chunk: Document) -> str:
    """把解析器的细粒度类型归一为 text/table 两类。"""
    raw = str(chunk.metadata.get("doc_type") or "").casefold()
    if "table" in raw or bool(re.search(r"(?m)^\s*\|.+\|\s*$", chunk.page_content)):
        return "table"
    return "text"


def _get_parser(chunk: Document) -> str:
    return str(chunk.metadata.get("parser") or "unknown").strip() or "unknown"


def _get_page(chunk: Document) -> int | None:
    value = chunk.metadata.get("page", chunk.metadata.get("source_page"))
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _is_noise_chunk(text: str) -> bool:
    """粗过滤：命中免责/声明类关键词则视为噪声 chunk，跳过 LLM 调用。"""
    return bool(_NOISE_PATTERNS.search(text[:600]))


# ── 语义去重：模块级模型缓存，避免多次调用重复加载 ──────────────────────────
_DEDUP_MODEL = None
_DEDUP_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def _get_dedup_model():
    global _DEDUP_MODEL
    if _DEDUP_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"[dedup] 加载语义去重模型：{_DEDUP_MODEL_NAME} ...")
        _DEDUP_MODEL = SentenceTransformer(_DEDUP_MODEL_NAME)
    return _DEDUP_MODEL


def _deduplicate(records: list[dict], threshold: float = 0.92) -> list[dict]:
    """基于语义向量的去重：对 question 字段计算余弦相似度，过滤重复问题。

    算法：
      1. 批量编码所有 question，得到归一化向量矩阵（N × D）。
      2. 矩阵自乘 embeddings @ embeddings.T 一次性得到余弦相似度矩阵（N × N）。
      3. 贪心遍历：当前问题与已保留集合中任意问题的相似度 >= threshold，则丢弃。

    Args:
        records:   输入记录列表，每条须含 "question" 键。
        threshold: 余弦相似度阈值，默认 0.92；越高则保留越多相似问题。

    Returns:
        去重后的记录列表，顺序与原列表一致，类型不变。
    """
    if len(records) <= 1:
        return list(records)

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        print(
            "[dedup] ⚠ sentence-transformers 未安装，回退到前缀去重。\n"
            "        请执行：pip install sentence-transformers"
        )
        seen: set[str] = set()
        out = []
        for r in records:
            key = r["question"][:10]
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out

    model = _get_dedup_model()
    questions = [r["question"] for r in records]

    embeddings: "np.ndarray" = model.encode(
        questions,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )  # shape: (N, D)

    sim_matrix: "np.ndarray" = embeddings @ embeddings.T  # shape: (N, N)

    kept: list[int] = []
    for i in range(len(records)):
        if not kept:
            kept.append(i)
            continue
        if sim_matrix[i, kept].max() < threshold:
            kept.append(i)

    return [records[i] for i in kept]


# ── 核心函数 ──────────────────────────────────────────────────────────────────

def load_all_chunks(db_path: Path) -> list[Document]:
    """通过 RAG 管理服务读取全量 chunks，不加载 embedding 模型。"""
    import asyncio
    if os.getenv("KNOWLEDGE_SERVICE_URL", "").strip():
        from react_agent.rag.runtime import create_configured_rag_runtime

        async def read_remote_chunks():
            runtime = create_configured_rag_runtime()
            try:
                return await runtime.get_admin_service().read_chunks()
            finally:
                await runtime.close()

        stored_chunks = asyncio.run(read_remote_chunks())
        source_label = os.environ["KNOWLEDGE_SERVICE_URL"]
    else:
        from react_agent.rag.runtime.offline import create_admin_service

        stored_chunks = create_admin_service(
            chroma_dir=str(db_path)
        ).read_chunks_sync()
        source_label = str(db_path)

    print(f"[generator] 通过 RAG 管理服务读取 chunks：{source_label}")
    all_docs = [
        Document(
            page_content=chunk.content,
            metadata={**dict(chunk.metadata), "chunk_id": chunk.chunk_id},
        )
        for chunk in stored_chunks
    ]
    print(f"[generator] 共读取 {len(all_docs)} 个 chunks")
    return all_docs


def stratified_sample(
    chunks: list[Document],
    max_chunks: int,
    seed: int,
) -> list[Document]:
    """
    按行业、正文/表格、解析器分层，并在每层内轮转文档。

    过滤规则：
      - chunk 文本过短（< 100 字）
      - 命中免责/声明类关键词（_is_noise_chunk）
    这样可避免大文档或高 chunk 数行业垄断评测集。
    """
    rng = random.Random(seed)

    valid = [
        c for c in chunks
        if len(c.page_content.strip()) >= 100 and not _is_noise_chunk(c.page_content)
    ]
    noise_count = len(chunks) - len(valid)
    if noise_count:
        print(f"[generator] 预过滤：跳过 {noise_count} 个噪声/过短 chunks")

    if max_chunks <= 0 or not valid:
        return []

    strata: dict[tuple[str, str, str], dict[str, list[Document]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for chunk in valid:
        source = str(
            chunk.metadata.get("source_file")
            or chunk.metadata.get("source")
            or chunk.metadata.get("chunk_id")
            or "unknown"
        ).replace("\\", "/")
        key = (_get_industry(chunk), _get_doc_type(chunk), _get_parser(chunk))
        strata[key][source].append(chunk)

    queues: dict[tuple[str, str, str], deque[Document]] = {}
    for key, by_source in strata.items():
        source_names = list(by_source)
        rng.shuffle(source_names)
        for docs in by_source.values():
            rng.shuffle(docs)
        queue: deque[Document] = deque()
        while source_names:
            remaining: list[str] = []
            for source in source_names:
                docs = by_source[source]
                if docs:
                    queue.append(docs.pop())
                if docs:
                    remaining.append(source)
            source_names = remaining
        queues[key] = queue

    active = list(queues)
    rng.shuffle(active)
    sampled: list[Document] = []
    while active and len(sampled) < min(max_chunks, len(valid)):
        remaining_keys: list[tuple[str, str, str]] = []
        for key in active:
            queue = queues[key]
            if queue and len(sampled) < max_chunks:
                sampled.append(queue.popleft())
            if queue:
                remaining_keys.append(key)
        active = remaining_keys

    distribution = Counter(
        (_get_industry(chunk), _get_doc_type(chunk), _get_parser(chunk))
        for chunk in sampled
    )
    for (industry, doc_type, parser), count in sorted(distribution.items()):
        print(
            f"  分层「{industry}/{doc_type}/{parser}」：抽 {count} chunks"
        )
    document_count = len(
        {
            str(chunk.metadata.get("source_file") or chunk.metadata.get("source"))
            for chunk in sampled
        }
    )
    print(
        f"[generator] 最终采样 {len(sampled)} 个 chunks，"
        f"覆盖 {document_count} 份文档"
    )
    return sampled


def build_evidence_units(
    chunks: list[Document],
    *,
    multi_chunk_ratio: float,
    seed: int,
    corpus: list[Document] | None = None,
) -> list[EvidenceUnit]:
    """把同一文档相邻页的 chunk 配成多证据单元，其余保持单 chunk。"""
    ratio = max(0.0, min(float(multi_chunk_ratio), 0.5))
    available = [
        chunk
        for chunk in (corpus or chunks)
        if len(chunk.page_content.strip()) >= 100
        and not _is_noise_chunk(chunk.page_content)
    ]
    by_source: dict[str, list[Document]] = defaultdict(list)
    for chunk in available:
        source = str(
            chunk.metadata.get("source_file")
            or chunk.metadata.get("source")
            or "unknown"
        ).replace("\\", "/")
        by_source[source].append(chunk)

    candidates: list[tuple[Document, Document]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for anchor in chunks:
        source = str(
            anchor.metadata.get("source_file")
            or anchor.metadata.get("source")
            or "unknown"
        ).replace("\\", "/")
        ordered = sorted(
            by_source.get(source, []),
            key=lambda item: (
                _get_page(item) if _get_page(item) is not None else 10**9,
                str(item.metadata.get("chunk_id") or ""),
            ),
        )
        anchor_id = str(anchor.metadata.get("chunk_id") or id(anchor))
        anchor_index = next(
            (
                index
                for index, item in enumerate(ordered)
                if str(item.metadata.get("chunk_id") or id(item)) == anchor_id
            ),
            None,
        )
        if anchor_index is None:
            continue
        for neighbor_index in (anchor_index - 1, anchor_index + 1):
            if neighbor_index < 0 or neighbor_index >= len(ordered):
                continue
            neighbor = ordered[neighbor_index]
            neighbor_id = str(neighbor.metadata.get("chunk_id") or id(neighbor))
            pages = sorted(
                page
                for page in (_get_page(anchor), _get_page(neighbor))
                if page is not None
            )
            pair_key = tuple(sorted((anchor_id, neighbor_id)))
            if (
                len(pages) != 2
                or pages[1] - pages[0] > 1
                or pair_key in seen_pairs
            ):
                continue
            seen_pairs.add(pair_key)
            pair = tuple(
                sorted(
                    (anchor, neighbor),
                    key=lambda item: (
                        _get_page(item)
                        if _get_page(item) is not None
                        else 10**9,
                        str(item.metadata.get("chunk_id") or ""),
                    ),
                )
            )
            candidates.append((pair[0], pair[1]))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    target = min(round(len(chunks) * ratio), len(candidates))
    used: set[str] = set()
    units: list[EvidenceUnit] = []
    for left, right in candidates:
        if len(units) >= target:
            break
        ids = {
            str(left.metadata.get("chunk_id") or id(left)),
            str(right.metadata.get("chunk_id") or id(right)),
        }
        if used.intersection(ids):
            continue
        units.append(EvidenceUnit((left, right), "multi_chunk"))
        used.update(ids)
        if sum(unit.scope == "multi_chunk" for unit in units) >= target:
            break

    for chunk in chunks:
        chunk_id = str(chunk.metadata.get("chunk_id") or id(chunk))
        if chunk_id not in used:
            units.append(EvidenceUnit((chunk,), "single_chunk"))

    rng.shuffle(units)
    multi_count = sum(unit.scope == "multi_chunk" for unit in units)
    print(
        f"[generator] 证据单元：{len(units)} 组"
        f"（multi={multi_count}, single={len(units) - multi_count}）"
    )
    return units


def _compact_text(value: str) -> str:
    return re.sub(r"[\s,，]", "", str(value or "")).casefold()


def _answer_numbers(value: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", _compact_text(value)))


def validate_qa_pair(
    pair: QAPair,
    unit: EvidenceUnit,
) -> tuple[bool, tuple[str, ...]]:
    """确定性检查引文归属、多 chunk 覆盖和答案数字是否有原文依据。"""
    reasons: list[str] = []
    question = pair.question.strip()
    answer = pair.ground_truth.strip()
    if len(question) < 6:
        reasons.append("question_too_short")
    if re.search(r"(该公司|该行业|本项目|上述公司|这家公司)", question):
        reasons.append("ambiguous_question")
    if len(answer) < 2:
        reasons.append("answer_too_short")
    if not pair.evidence:
        reasons.append("missing_evidence_quotes")

    cited_indices: set[int] = set()
    for evidence in pair.evidence:
        index = evidence.chunk_index - 1
        quote = _compact_text(evidence.quote)
        if index < 0 or index >= len(unit.chunks):
            reasons.append("invalid_evidence_chunk_index")
            continue
        if len(quote) < 6:
            reasons.append("evidence_quote_too_short")
            continue
        cited_indices.add(index)
        if quote not in _compact_text(unit.chunks[index].page_content):
            reasons.append("evidence_quote_not_found")

    if unit.scope == "multi_chunk" and len(cited_indices) < 2:
        reasons.append("multi_chunk_evidence_incomplete")

    corpus = _compact_text("\n".join(chunk.page_content for chunk in unit.chunks))
    missing_numbers = sorted(
        number for number in _answer_numbers(answer) if number not in corpus
    )
    if missing_numbers:
        reasons.append("unsupported_answer_numbers:" + ",".join(missing_numbers))

    unique_reasons = tuple(dict.fromkeys(reasons))
    return not unique_reasons, unique_reasons


def _render_evidence_blocks(unit: EvidenceUnit, data_dir: Path) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(unit.chunks, 1):
        source = _get_rel_src(chunk, data_dir)
        page = _get_page(chunk)
        blocks.append(
            f"[片段 {index} | {source} | 页码 {page if page is not None else '未知'}"
            f" | 类型 {_get_doc_type(chunk)}]\n{chunk.page_content[:1600]}"
        )
    return "\n\n".join(blocks)


def _gold_sources(unit: EvidenceUnit, data_dir: Path) -> list[dict]:
    pages_by_source: dict[str, set[int]] = defaultdict(set)
    for chunk in unit.chunks:
        source = _get_rel_src(chunk, data_dir)
        page = _get_page(chunk)
        if page is not None:
            pages_by_source[source].add(page)
        else:
            pages_by_source[source]
    return [
        {"source_file": source, "pages": sorted(pages)}
        for source, pages in pages_by_source.items()
    ]


def _evidence_payload(pair: QAPair, unit: EvidenceUnit) -> list[dict]:
    payload: list[dict] = []
    for evidence in pair.evidence:
        index = evidence.chunk_index - 1
        if 0 <= index < len(unit.chunks):
            payload.append(
                {
                    "chunk_id": str(
                        unit.chunks[index].metadata.get("chunk_id") or ""
                    ),
                    "quote": evidence.quote.strip(),
                }
            )
    return payload
class _AdaptiveWorkerPool:
    def __init__(self, initial: int, min_workers: int = 1):
        self._initial = initial
        self._workers = initial
        self._min = min_workers
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(initial)
        self._pending_reduction = 0

    def acquire(self):
        self._sem.acquire()

    def release(self):
        with self._lock:
            if self._pending_reduction > 0:
                self._pending_reduction -= 1
            else:
                self._sem.release()

    @property
    def workers(self) -> int:
        return self._workers

    def on_rate_limited(self):
        with self._lock:
            if self._workers > self._min:
                self._workers = max(self._min, self._workers - 1)
                self._pending_reduction += 1
                print(f"[adaptive] 触发限流，并发数降至 {self._workers}")

    def on_success(self):
        with self._lock:
            if self._workers < self._initial:
                self._workers = min(self._initial, self._workers + 1)
                self._sem.release()


# ★ 核心改造 Step 4：改造 LLM 调用函数
#
# 变化对比：
#   旧版：接收原始 llm，返回 str | None，调用方还要再解析字符串
#   新版：接收 structured_llm，返回 QAList | None，调用方直接用对象
#
# 注意 resp 的处理：
#   旧版需要 resp.content（取原始文本）
#   新版直接 return resp（resp 已经是 QAList 对象，框架自动完成了解析）
def _call_llm_with_retry(
    structured_llm,
    prompt: str,
    pool: "_AdaptiveWorkerPool",
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> BaseModel | None:
    """带指数退避重试的 LLM 调用，感知限流错误并通知 pool 动态调整并发。"""
    for attempt in range(max_retries):
        try:
            resp = structured_llm.invoke(prompt)
            pool.on_success()
            return resp
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "503" in err_str
            if is_rate_limit:
                pool.on_rate_limited()
                wait = backoff_base ** attempt * 3
            else:
                wait = backoff_base ** attempt

            if attempt < max_retries - 1:
                print(f"    ↻ LLM 调用失败（第 {attempt+1} 次），{wait:.0f}s 后重试：{e}")
                sleep(wait)
            else:
                print(f"    ✗ LLM 调用失败（已重试 {max_retries} 次），放弃：{e}")
    return None


def _with_structured_output(llm, schema: type[BaseModel]):
    """为 DeepSeek 结构化生成关闭 thinking，避免 tool_choice 冲突。"""
    api_base = str(getattr(llm, "openai_api_base", "") or "").casefold()
    model_name = str(getattr(llm, "model_name", "") or "").casefold()
    structured = llm.with_structured_output(schema, method="function_calling")
    if "deepseek" in api_base or "deepseek" in model_name:
        structured = structured.bind(
            extra_body={"thinking": {"type": "disabled"}}
        )
    return structured


def _process_unit(
    args: tuple,
) -> tuple[int, list[dict], Counter[str]]:
    """处理单/多 chunk 证据单元，并返回验证通过的候选记录。"""
    i, unit, data_dir, n_per_unit, structured_llm, max_retries, pool = args
    primary = unit.chunks[0]
    industry = _get_industry(primary)
    rel_src = _get_rel_src(primary, data_dir)
    page = _get_page(primary)
    chunk_ids = [
        str(chunk.metadata.get("chunk_id") or "").strip()
        for chunk in unit.chunks
    ]
    chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
    scope_instruction = (
        "当前是多片段任务：每个问题必须同时结合至少两个片段才能完整回答，"
        "并至少提供来自两个不同 chunk_index 的引文。"
        if unit.scope == "multi_chunk"
        else "当前是单片段任务：问题必须能仅依据片段 1 完整回答。"
    )

    prompt = _QA_PROMPT.format(
        n=n_per_unit,
        evidence_blocks=_render_evidence_blocks(unit, data_dir),
        scope_instruction=scope_instruction,
    )

    response = _call_llm_with_retry(
        structured_llm,
        prompt,
        pool,
        max_retries=max_retries,
    )
    if not isinstance(response, QAList):
        return i, [], Counter({"llm_failure": 1})

    records: list[dict] = []
    rejected: Counter[str] = Counter()
    for pair in response.pairs:
        if len(records) >= n_per_unit:
            break
        valid, reasons = validate_qa_pair(pair, unit)
        if not valid:
            rejected.update(reasons)
            continue
        case_digest = hashlib.sha256(
            f"{'|'.join(chunk_ids)}\0{pair.question}".encode("utf-8")
        ).hexdigest()[:16]
        sources = _gold_sources(unit, data_dir)
        records.append(
            {
                "schema_version": 3,
                "case_id": f"rag_{case_digest}",
                "question": pair.question.strip(),
                "ground_truth": pair.ground_truth.strip(),
                "answerable": True,
                "gold_chunk_ids": chunk_ids,
                "gold_sources": sources,
                "source_file": rel_src,
                "page": page,
                "industry": industry,
                "category": pair.category,
                "reasoning_scope": unit.scope,
                "content_types": sorted(
                    {_get_doc_type(chunk) for chunk in unit.chunks}
                ),
                "parsers": sorted({_get_parser(chunk) for chunk in unit.chunks}),
                "gold_evidence": _evidence_payload(pair, unit),
                "evidence_validation": {
                    "status": "passed",
                    "checks": [
                        "exact_quote",
                        "answer_number_support",
                        *(
                            ["multi_chunk_coverage"]
                            if unit.scope == "multi_chunk"
                            else []
                        ),
                    ],
                },
                "review_status": "pending",
                "review_notes": "",
                "chunk_text": "\n\n".join(
                    chunk.page_content[:1000] for chunk in unit.chunks
                ),
            }
        )
    return i, records, rejected


def generate_qa_pairs(
    units: list[EvidenceUnit],
    data_dir: Path,
    n_per_chunk: int,
    llm,
    verbose: bool = True,
    max_workers: int = 4,
    max_retries: int = 3,
) -> list[dict]:
    """
    并发调用 LLM 为每个证据单元生成 QA 对，返回验证并去重后的记录。

    Args:
        max_workers: 并发线程数，受限于 LLM API 速率限制，建议 2-8。
        max_retries: 单个 chunk 最大重试次数。
    """

    structured_llm = _with_structured_output(llm, QAList)

    total = len(units)
    all_records: list[dict] = []
    completed = 0
    rejected: Counter[str] = Counter()

    pool = _AdaptiveWorkerPool(initial=max_workers)

    task_args = [
        (i, unit, data_dir, n_per_chunk, structured_llm, max_retries, pool)
        for i, unit in enumerate(units)
    ]

    def _submit(arg):
        pool.acquire()
        try:
            return _process_unit(arg)
        finally:
            pool.release()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_submit, arg): arg[0] for arg in task_args}

        for future in as_completed(futures):
            unit_idx, records, unit_rejected = future.result()
            all_records.extend(records)
            rejected.update(unit_rejected)
            completed += 1

            if verbose:
                unit = units[unit_idx]
                rel_src = _get_rel_src(unit.chunks[0], data_dir)
                page = _get_page(unit.chunks[0])
                pct = completed / total * 100
                print(
                    f"  [{completed:>3}/{total}] {rel_src}:p{page}"
                    f"  {unit.scope} 通过 {len(records)} 条  ({pct:.0f}%)"
                    f"  [并发: {pool.workers}]"
                )

    before = len(all_records)
    all_records = _deduplicate(all_records)
    all_records.sort(key=lambda record: record["case_id"])
    print(f"[generator] 去重：{before} → {len(all_records)} 条 QA 对")
    if rejected:
        print(f"[generator] 自动证据校验淘汰：{dict(rejected)}")
    return all_records


def generate_no_answer_cases(
    chunks: list[Document],
    data_dir: Path,
    count: int,
    llm,
    *,
    seed: int,
    max_retries: int,
) -> list[dict]:
    """生成待全库筛查和人工确认的无答案候选问题。"""
    if count <= 0 or not chunks:
        return []

    rng = random.Random(seed)
    evidence_chunks = list(chunks)
    rng.shuffle(evidence_chunks)
    evidence_chunks = evidence_chunks[: min(12, len(evidence_chunks))]
    blocks = []
    for index, chunk in enumerate(evidence_chunks, 1):
        blocks.append(
            f"[主题 {index} | {_get_rel_src(chunk, data_dir)}"
            f" | {_get_industry(chunk)}]\n{chunk.page_content[:600]}"
        )

    structured_llm = _with_structured_output(llm, NoAnswerList)
    pool = _AdaptiveWorkerPool(initial=1)
    response = _call_llm_with_retry(
        structured_llm,
        _NO_ANSWER_PROMPT.format(
            n=count,
            evidence_blocks="\n\n".join(blocks),
        ),
        pool,
        max_retries=max_retries,
    )
    if not isinstance(response, NoAnswerList):
        print("[generator] 无答案候选生成失败，跳过")
        return []

    records: list[dict] = []
    seen: set[str] = set()
    for item in response.cases:
        question = item.question.strip()
        normalized = _compact_text(question)
        if (
            len(question) < 6
            or normalized in seen
            or re.search(r"(该公司|该行业|本项目|上述公司|这家公司)", question)
        ):
            continue
        seen.add(normalized)
        case_digest = hashlib.sha256(
            f"no_answer\0{question}".encode("utf-8")
        ).hexdigest()[:16]
        records.append(
            {
                "schema_version": 3,
                "case_id": f"rag_noanswer_{case_digest}",
                "question": question,
                "ground_truth": "知识库未提供足够信息，应明确说明无法确定。",
                "answerable": False,
                "gold_chunk_ids": [],
                "gold_sources": [],
                "source_file": "",
                "page": None,
                "industry": "跨行业",
                "category": item.category,
                "reasoning_scope": "no_answer",
                "content_types": [],
                "parsers": [],
                "gold_evidence": [],
                "evidence_validation": {
                    "status": "needs_review",
                    "reason": item.rationale.strip(),
                    "required_checks": [
                        "full_corpus_retrieval",
                        "human_confirmation",
                    ],
                },
                "review_status": "pending",
                "review_notes": "",
                "chunk_text": "",
            }
        )
        if len(records) >= count:
            break
    records.sort(key=lambda record: record["case_id"])
    print(f"[generator] 生成 {len(records)} 条无答案候选（需人工确认）")
    return records


def save_dataset(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist: dict[str, int] = defaultdict(int)
    for r in records:
        dist[r["industry"]] += 1
    print(f"\n[generator] 数据集已保存：{output_path}  共 {len(records)} 条")
    print("  行业分布：")
    for ind, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {ind:12s} {cnt:>4} 条")


def balanced_subset(
    records: list[dict],
    size: int,
    *,
    seed: int,
) -> list[dict]:
    """按可回答性、行业、类别和推理范围轮转抽取稳定子集。"""
    if size <= 0 or size >= len(records):
        return list(records)

    rng = random.Random(seed)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            bool(record.get("answerable", True)),
            str(record.get("industry") or "未知"),
            str(record.get("category") or "unknown"),
            str(record.get("reasoning_scope") or "unknown"),
        )
        buckets[key].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    keys = list(buckets)
    rng.shuffle(keys)
    selected: list[dict] = []
    while keys and len(selected) < size:
        remaining: list[tuple] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < size:
                selected.append(bucket.pop())
            if bucket:
                remaining.append(key)
        keys = remaining
    return sorted(selected, key=lambda record: record["case_id"])


def save_dataset_bundle(
    records: list[dict],
    output_path: Path,
    *,
    smoke_size: int,
    regression_size: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Path]:
    """导出 full candidate、smoke、regression 和清单文件。"""
    suffix = output_path.suffix or ".jsonl"
    stem = output_path.stem
    paths = {
        "full": output_path,
        "smoke": output_path.with_name(f"{stem}.smoke{suffix}"),
        "regression": output_path.with_name(f"{stem}.regression{suffix}"),
        "manifest": output_path.with_name(f"{stem}.manifest.json"),
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"输出已存在：{joined}；确认后使用 --overwrite 覆盖"
        )

    smoke = balanced_subset(records, smoke_size, seed=seed + 1)
    regression = balanced_subset(records, regression_size, seed=seed + 2)
    save_dataset(records, paths["full"])
    save_dataset(smoke, paths["smoke"])
    save_dataset(regression, paths["regression"])

    manifest = {
        "schema_version": 1,
        "dataset_schema_version": 3,
        "seed": seed,
        "counts": {
            "full": len(records),
            "smoke": len(smoke),
            "regression": len(regression),
            "answerable": sum(
                record.get("answerable") is not False for record in records
            ),
            "no_answer": sum(
                record.get("answerable") is False for record in records
            ),
            "multi_chunk": sum(
                record.get("reasoning_scope") == "multi_chunk"
                for record in records
            ),
            "pending_review": sum(
                record.get("review_status") == "pending" for record in records
            ),
        },
        "files": {
            key: str(path)
            for key, path in paths.items()
            if key != "manifest"
        },
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    with paths["manifest"].open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"[generator] 数据清单：{paths['manifest']}")
    return paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="金融研报 RAG 评测数据集生成器")
    p.add_argument("--data_dir", default=str(_SRC / "data"), help="文档根目录")
    p.add_argument(
        "--output",
        default=str(
            _SRC
            / "eval"
            / "results"
            / "eval_dataset_docling_v1.candidate.jsonl"
        ),
        help="full 候选集输出路径；同时导出 smoke/regression/manifest",
    )
    p.add_argument("--n_per_chunk", type=int, default=1, help="每个证据单元生成 QA 数")
    p.add_argument("--max_chunks", type=int, default=150, help="最大采样 chunk 数")
    p.add_argument("--multi_chunk_ratio", type=float, default=0.25)
    p.add_argument("--no_answer_count", type=int, default=20)
    p.add_argument("--smoke_size", type=int, default=30)
    p.add_argument("--regression_size", type=int, default=120)
    p.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    p.add_argument(
        "--model",
        default="deepseek/deepseek-v4-flash",
        help="LLM，格式 provider/model",
    )
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--verbose", action="store_true", help="打印每条生成进度")
    p.add_argument("--dry_run", action="store_true", help="只读语料并展示抽样，不调用 LLM")
    p.add_argument("--overwrite", action="store_true", help="允许覆盖已有候选输出")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve()

    if not data_dir.exists() and not os.getenv("KNOWLEDGE_SERVICE_URL", "").strip():
        print(f"[error] data_dir 不存在：{data_dir}")
        sys.exit(1)

    chunks = load_all_chunks(settings.tools.vector_store.CHROMA_DB_PATH)
    sampled = stratified_sample(chunks, args.max_chunks, args.seed)
    units = build_evidence_units(
        sampled,
        multi_chunk_ratio=args.multi_chunk_ratio,
        seed=args.seed,
        corpus=chunks,
    )
    if args.dry_run:
        print("[generator] dry-run 完成：未调用 LLM、未写入数据集")
        return

    llm = load_chat_model(args.model)
    print(f"[generator] 使用 LLM：{llm}")

    answerable_records = generate_qa_pairs(
        units,
        data_dir,
        args.n_per_chunk,
        llm,
        verbose=args.verbose,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )
    if not answerable_records:
        raise RuntimeError(
            "No answerable QA records passed validation; no files were written."
        )
    no_answer_records = generate_no_answer_cases(
        sampled,
        data_dir,
        args.no_answer_count,
        llm,
        seed=args.seed + 1000,
        max_retries=args.max_retries,
    )
    records = sorted(
        [*answerable_records, *no_answer_records],
        key=lambda record: record["case_id"],
    )
    save_dataset_bundle(
        records,
        output_path,
        smoke_size=args.smoke_size,
        regression_size=args.regression_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print("\n✅ 候选数据集生成完成；全部记录仍需人工审核")


if __name__ == "__main__":
    main()
