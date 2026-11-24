"""本地 PDF 解析路径。

集中容纳金融研报版面解析算法、统一解析契约适配和页数检查，避免一条本地
PDF 路径被拆成多个只有单一调用方的小模块。
"""

# 金融研报级 PDF 解析：
#   - PyMuPDF  → 带坐标文字块提取、多栏重排、页眉页脚过滤
#   - pdfplumber → 结构化表格提取（转 Markdown，整块保留）
#   - 降级兜底  → 两库均缺失时自动回落到 PyPDFLoader

from __future__ import annotations

import asyncio
import logging
import re
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.ingestion.contracts import ParseRequest, ParseResult, ParsedChunk
from knowledge.ingestion.parsing import source_identity

# ──────────────────────────────────────────────
# 内部常量
# ──────────────────────────────────────────────
_HEADER_FOOTER_RATIO = 0.07   # 页面顶部 / 底部各 7% 视为页眉页脚
_COL_GAP_RATIO = 0.12   # 双栏之间空白带宽度阈值（相对页宽）
_TABLE_OVERLAP_THRESH = 0.4   # 文字块与表格区域 IoU 超过此值则视为表格内文字
_MIN_BLOCK_CHARS = 3      # 过滤掉字符数极少的噪声块（如单个页码）
_COORD_TOLERANCE = 2.0   # 允许的浮点舍入误差（单位：pt）
_DEFAULT_CHUNK_MAX_TOKENS = 448
_DEFAULT_CHUNK_OVERLAP_TOKENS = 48
_DEFAULT_CHUNK_TOKENIZER = "BAAI/bge-small-zh-v1.5"


logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_chunk_tokenizer(model_name: str) -> Any:
    """按进程惰性加载本地 PDF 分块 Tokenizer，避免导入阶段加载模型。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _create_text_splitter(
    *,
    tokenizer_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[RecursiveCharacterTextSplitter, Any]:
    if not tokenizer_name.strip():
        raise ValueError("本地 PDF 分块 Tokenizer 不能为空")
    if chunk_size < 64:
        raise ValueError("本地 PDF chunk_size 不能小于 64 Token")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("本地 PDF chunk_overlap 必须大于等于 0 且小于 chunk_size")

    tokenizer = _load_chunk_tokenizer(tokenizer_name.strip())
    tokenizer_limit = _tokenizer_limit(tokenizer)
    if tokenizer_limit is not None and chunk_size + 2 > tokenizer_limit:
        raise ValueError(
            "本地 PDF chunk_size 必须为 Embedding 特殊 Token 预留空间："
            f"chunk_size={chunk_size}, model_max_length={tokenizer_limit}"
        )
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 中文研报常见语义边界：段落 > 句号 > 分号 > 逗号 > 空格。
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter, tokenizer


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def _tokenizer_limit(tokenizer: Any) -> int | None:
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    # Transformers 用极大整数表达“未声明上限”，不能把它当作真实模型窗口。
    return value if 0 < value < 1_000_000 else None


# ──────────────────────────────────────────────
# 依赖检测
# ──────────────────────────────────────────────
def _check_deps() -> Tuple[bool, bool]:
    """返回 (has_fitz, has_pdfplumber)"""
    try:
        import fitz  # noqa: F401
        has_fitz = True
    except ImportError:
        has_fitz = False

    try:
        import pdfplumber  # noqa: F401
        has_pdfplumber = True
    except ImportError:
        has_pdfplumber = False

    return has_fitz, has_pdfplumber


# ──────────────────────────────────────────────
# 表格工具
# ──────────────────────────────────────────────
def _table_to_markdown(table: list) -> str:
    """
    将 pdfplumber 返回的二维列表转为 Markdown 表格字符串。
    自动处理 None 单元格和合并列情况。
    """
    if not table or not table[0]:
        return ""

    def _clean(cell) -> str:
        if cell is None:
            return ""
        return re.sub(r"\s+", " ", str(cell)).strip()

    header = table[0]
    col_count = len(header)
    rows = table[1:]

    # 修复行宽不一致（pdfplumber 偶发）
    def _pad(row: list) -> list:
        row = list(row)
        if len(row) < col_count:
            row += [""] * (col_count - len(row))
        return row[:col_count]

    md_lines = []
    md_lines.append("| " + " | ".join(_clean(c) for c in header) + " |")
    md_lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for row in rows:
        md_lines.append("| " + " | ".join(_clean(c) for c in _pad(row)) + " |")

    return "\n".join(md_lines)


def _get_table_bboxes(tables: list) -> List[Tuple[float, float, float, float]]:
    """从已检测的表格获取 bbox，避免对同一页重复运行检测。"""
    return [
        (table.bbox[0], table.bbox[1], table.bbox[2], table.bbox[3])
        for table in tables
    ]


def _overlap_ratio(b1: tuple, b2: tuple) -> float:
    """计算两个 bbox 的交集面积 / b1 面积（用于判断 b1 是否在 b2 内部）。"""
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_b1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    return inter / area_b1


def _is_in_table(block_bbox: tuple,
                 table_bboxes: List[tuple],
                 thresh: float = _TABLE_OVERLAP_THRESH) -> bool:
    return any(_overlap_ratio(block_bbox, tb) >= thresh for tb in table_bboxes)


# ──────────────────────────────────────────────
# 版面分析工具
# ──────────────────────────────────────────────
def _filter_header_footer(
    blocks: list,
    page_height: float,
    margin_ratio: float = _HEADER_FOOTER_RATIO,
) -> list:
    """过滤页眉（顶部）、页脚（底部）及字符数极少的噪声块。"""
    top_limit = page_height * margin_ratio
    bot_limit = page_height * (1 - margin_ratio)

    kept = []
    for b in blocks:
        y0, y1 = b["bbox"][1], b["bbox"][3]
        text = b.get("text", "").strip()
        if y0 < top_limit or y1 > bot_limit:
            continue
        if len(text) < _MIN_BLOCK_CHARS:
            continue
        kept.append(b)
    return kept


def _detect_and_sort_columns(blocks: list, page_width: float) -> list:
    """
    检测单栏 / 双栏布局，并按正确阅读顺序重排 blocks。

    双栏判定逻辑：
      1. 统计所有块的 x 起点，若左右两侧均有块且中间存在明显空白带
         （宽度 > page_width * _COL_GAP_RATIO），则视为双栏。
      2. 双栏时：先按 y 排序左栏，再按 y 排序右栏，左栏内容在前。
    """
    if not blocks:
        return blocks

    mid = page_width / 2
    left_blocks = [b for b in blocks if b["bbox"][0] < mid]
    right_blocks = [b for b in blocks if b["bbox"][0] >= mid]

    # 双栏判断：两侧均有内容，且中间有明显空白带
    is_two_col = False
    if left_blocks and right_blocks:
        left_x1_max = max(b["bbox"][2] for b in left_blocks)
        right_x0_min = min(b["bbox"][0] for b in right_blocks)
        gap = right_x0_min - left_x1_max
        if gap > page_width * _COL_GAP_RATIO:
            is_two_col = True

    if is_two_col:
        # 各栏内按 y 坐标（行号）排序
        left_sorted = sorted(left_blocks,  key=lambda b: b["bbox"][1])
        right_sorted = sorted(right_blocks, key=lambda b: b["bbox"][1])
        return left_sorted + right_sorted
    else:
        # 单栏：整体按 y 排序即可
        return sorted(blocks, key=lambda b: b["bbox"][1])


# ──────────────────────────────────────────────
# PyMuPDF 文字提取
# ──────────────────────────────────────────────
def _extract_text_blocks_fitz(fitz_page, table_bboxes: list) -> List[dict]:
    """
    用 PyMuPDF 提取文字块，返回 [{text, bbox}, ...] 列表。
    已排除落在表格区域内的块（由 pdfplumber 单独处理）。

    注意：pdfplumber 的 y 轴原点在页面顶部（PDF 标准：底部），
    fitz 的 bbox 也是顶部原点，两者一致，可直接比较。
    """
    import fitz

    raw_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    result = []
    for block in raw_dict.get("blocks", []):
        if block.get("type") != 0:          # type=1 是图片块，跳过
            continue
        bbox = tuple(block["bbox"])          # (x0, y0, x1, y1)
        if _is_in_table(bbox, table_bboxes):
            continue                         # 表格内文字交给 pdfplumber

        lines_text = []
        for line in block.get("lines", []):
            line_str = "".join(s["text"] for s in line.get("spans", []))
            lines_text.append(line_str)
        text = "\n".join(lines_text).strip()

        if len(text) >= _MIN_BLOCK_CHARS:
            result.append({"text": text, "bbox": bbox})
    return result


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────
def load_pdf_with_layout(
    file_path: str,
    chunk_size: int = _DEFAULT_CHUNK_MAX_TOKENS,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP_TOKENS,
    tokenizer_name: str = _DEFAULT_CHUNK_TOKENIZER,
    source_root: str | None = None,
) -> List[Document]:
    """
    金融研报级 PDF 解析主函数。

    处理流程（每页）：
      1. pdfplumber  → 提取表格 bbox + 表格内容（转 Markdown Document）
      2. PyMuPDF     → 提取带坐标文字块，排除表格区域
      3. 版面分析    → 过滤页眉页脚、多栏重排
      4. 文字分块    → 与 Embedding 对齐的 Tokenizer + 中文友好分隔符
      5. 汇总        → 文字 chunk + 表格 Document 一并返回

    降级策略：
      - 仅缺 pdfplumber：跳过表格提取，继续文字解析（发出警告）
      - 仅缺 PyMuPDF  ：不应发生（依赖已在项目元数据声明，此处兜底）
      - 两者均缺      ：回落到 PyPDFLoader（发出明显警告）

    Returns:
        List[Document]，已完成分块，metadata 包含：
          source, page, chunk_id, doc_type ("text" | "table")
    """
    splitter, tokenizer = _create_text_splitter(
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    tokenizer_limit = _tokenizer_limit(tokenizer)
    has_fitz, has_pdfplumber = _check_deps()
    data_dir = source_root or str(Path(file_path).parent)
    try:
        filename = str(Path(file_path).relative_to(data_dir))
    except ValueError:
        filename = Path(file_path).name

    try:
        industry = Path(file_path).relative_to(data_dir).parts[0]
    except (ValueError, IndexError):
        industry = None

    # ── 降级：两者均缺 ──────────────────────────────────────────
    if not has_fitz:
        warnings.warn(
            f"⚠️  [local_pdf] PyMuPDF 未安装，'{filename}' 回落到 PyPDFLoader。\n"
            "    多栏/表格解析将失效，建议：pip install pymupdf pdfplumber",
            RuntimeWarning,
            stacklevel=2,
        )
        from langchain_community.document_loaders import PyPDFLoader

        return splitter.split_documents(PyPDFLoader(file_path).load())

    if not has_pdfplumber:
        warnings.warn(
            f"⚠️  [local_pdf] pdfplumber 未安装，'{filename}' 将跳过表格提取。\n"
            "    建议：pip install pdfplumber",
            RuntimeWarning,
            stacklevel=2,
        )

    import fitz

    all_docs: List[Document] = []
    fitz_doc = fitz.open(file_path)

    # 按需打开 pdfplumber（避免重复 open）
    plumber_ctx = None
    if has_pdfplumber:
        import pdfplumber
        try:
            plumber_ctx = pdfplumber.open(file_path)
        except Exception as e:
            warnings.warn(
                f"⚠️  [local_pdf] pdfplumber 打开 '{filename}' 失败，跳过表格提取：{e}",
                RuntimeWarning,
                stacklevel=2,
            )

    try:
        for page_num in range(len(fitz_doc)):
            human_page_num = page_num + 1
            fitz_page = fitz_doc[page_num]
            page_w = fitz_page.rect.width
            page_h = fitz_page.rect.height

            # ── Step 1：表格区域 & 表格内容（pdfplumber） ────────────
            table_bboxes: List[tuple] = []
            if plumber_ctx is not None:
                try:
                    plumber_page = plumber_ctx.pages[page_num]
                    plumber_h = plumber_page.height
                    found_tables = list(plumber_page.find_tables())

                    if abs(plumber_h - page_h) > _COORD_TOLERANCE:
                        warnings.warn(
                            f"⚠️  [local_pdf] '{filename}' 第 {human_page_num} 页坐标系不一致 "
                            f"(fitz={page_h:.1f}, plumber={plumber_h:.1f})，跳过表格区域剔除。",
                            RuntimeWarning, stacklevel=2,
                        )
                    else:
                        table_bboxes = _get_table_bboxes(found_tables)

                    for t_idx, table in enumerate(found_tables):
                        raw_table = table.extract()
                        md = _table_to_markdown(raw_table)
                        if not md.strip():
                            continue
                        table_tokens = _token_count(tokenizer, md)
                        exceeds_limit = bool(
                            tokenizer_limit is not None
                            and table_tokens > tokenizer_limit
                        )
                        if exceeds_limit:
                            logger.warning(
                                "[LOCAL_PDF] 完整表格超过 Embedding Token 上限 "
                                "source=%s page=%s table=%s tokens=%s limit=%s",
                                filename,
                                human_page_num,
                                t_idx,
                                table_tokens,
                                tokenizer_limit,
                            )
                        chunk_id = f"{filename}::page_{human_page_num}::table_{t_idx}"
                        all_docs.append(Document(
                            page_content=md,
                            metadata={
                                "source":   file_path,
                                "page":     human_page_num,
                                "chunk_id": chunk_id,
                                "doc_type": "table",
                                "num_tokens": table_tokens,
                                "embedding_token_limit_exceeded": exceeds_limit,
                                **({"industry": industry} if industry else {}),  # ← None 时不写入该字段
                            },
                        ))
                except Exception as e:
                    # 当前页表格提取失败，降级：清空 bbox，PyMuPDF 正文提取不受影响
                    warnings.warn(
                        f"⚠️  [local_pdf] '{filename}' 第 {human_page_num} 页表格提取失败，"
                        f"跳过本页表格（正文仍正常处理）：{e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    table_bboxes = []

            # ── Step 2：文字块提取（PyMuPDF，排除表格区域） ──────────
            raw_blocks = _extract_text_blocks_fitz(fitz_page, table_bboxes)

            # ── Step 3：版面分析 ──────────────────────────────────────
            blocks = _filter_header_footer(raw_blocks, page_h)
            blocks = _detect_and_sort_columns(blocks, page_w)

            if not blocks:
                continue

            # ── Step 4：合并文字 → 分块 ───────────────────────────────
            page_text = "\n".join(b["text"] for b in blocks).strip()
            if not page_text:
                continue

            text_chunks = splitter.split_text(page_text)
            for c_idx, chunk_text in enumerate(text_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                chunk_id = f"{filename}::page_{human_page_num}::chunk_{c_idx}"
                all_docs.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "source":   file_path,
                        "page":     human_page_num,
                        "chunk_id": chunk_id,
                        "doc_type": "text",
                        "num_tokens": _token_count(tokenizer, chunk_text),
                        **({"industry": industry} if industry else {}),  # ← None 时不写入该字段
                    },
                ))

    finally:
        fitz_doc.close()
        if plumber_ctx is not None:
            plumber_ctx.close()

    if not all_docs:
        warnings.warn(
            f"⚠️  [local_pdf] '{filename}' 解析后无任何内容，请检查文件是否为图片型 PDF。",
            RuntimeWarning,
            stacklevel=2,
        )

    return all_docs


class LocalPdfParserAdapter:
    """执行本地 PDF 解析并返回建库内部的统一契约。"""

    def __init__(
        self,
        *,
        chunking_max_tokens: int = _DEFAULT_CHUNK_MAX_TOKENS,
        chunking_overlap_tokens: int = _DEFAULT_CHUNK_OVERLAP_TOKENS,
        chunking_tokenizer: str = _DEFAULT_CHUNK_TOKENIZER,
    ) -> None:
        if chunking_max_tokens < 64:
            raise ValueError("chunking_max_tokens 不能小于 64")
        if not 0 <= chunking_overlap_tokens < chunking_max_tokens:
            raise ValueError(
                "chunking_overlap_tokens 必须大于等于 0 且小于 chunking_max_tokens"
            )
        if not chunking_tokenizer.strip():
            raise ValueError("chunking_tokenizer 不能为空")
        self._chunking_max_tokens = chunking_max_tokens
        self._chunking_overlap_tokens = chunking_overlap_tokens
        self._chunking_tokenizer = chunking_tokenizer.strip()

    async def parse(self, request: ParseRequest) -> ParseResult:
        normalized = request.normalized()
        return await asyncio.to_thread(self._parse_sync, normalized)

    def _parse_sync(self, request: ParseRequest) -> ParseResult:
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            documents = load_pdf_with_layout(
                request.file_path,
                source_root=request.source_root,
                chunk_size=self._chunking_max_tokens,
                chunk_overlap=self._chunking_overlap_tokens,
                tokenizer_name=self._chunking_tokenizer,
            )

        source_file, industry = source_identity(
            request.file_path,
            request.source_root,
        )
        chunks: list[ParsedChunk] = []
        for index, document in enumerate(documents or []):
            content = str(document.page_content or "").strip()
            if not content:
                continue
            metadata: dict[str, Any] = dict(document.metadata or {})
            page_value = metadata.get("page")
            try:
                page = int(page_value) if page_value is not None else None
            except (TypeError, ValueError):
                page = None
            chunks.append(
                ParsedChunk(
                    content=content,
                    source_file=source_file,
                    source_page=page,
                    source_pages=(page,) if page is not None else (),
                    chunk_id=str(
                        metadata.get("chunk_id")
                        or f"{source_file}::local_{index:06d}"
                    ),
                    doc_type=str(metadata.get("doc_type") or "text"),
                    parser_name="local",
                    metadata={
                        **metadata,
                        **({"industry": industry} if industry else {}),
                    },
                )
            )

        return ParseResult(
            source_file=source_file,
            parser_name="local",
            chunks=tuple(chunks),
            processing_time=time.perf_counter() - started,
            warnings=tuple(str(item.message) for item in captured),
        )

__all__ = ["LocalPdfParserAdapter"]


