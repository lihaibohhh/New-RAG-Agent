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

import hashlib
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

from langchain_core.documents import Document
from pydantic import BaseModel

from eval.dataset.models import (
    EvidenceQuote as EvidenceQuote,
    EvidenceUnit,
    NoAnswerCase as NoAnswerCase,
    NoAnswerList,
    QAPair,
    QAList,
)
from eval.dataset.sampling import (
    _get_doc_type,
    _get_industry,
    _get_page,
    _get_parser,
    _get_rel_src,
)
from eval.dataset.validation import (
    _compact_text,
    validate_qa_pair,
)


# ══════════════════════════════════════════════════════════════════════════════
# ★ 核心改造 Step 1：定义数据模型
#
# 工程直觉：把你"期望 LLM 输出的形状"，用 Python 类写出来。
# 这就是你和 LLM 之间的"合同"——它必须按这个格式填，不能乱发挥。
#
# Field(description=...) 的作用：这段描述会被翻译进 JSON Schema 里，
# 模型在生成时能读到它，相当于表格里的填写说明，比在 Prompt 里叮嘱更可靠。
# ══════════════════════════════════════════════════════════════════════════════

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
