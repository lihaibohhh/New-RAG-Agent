"""知识库本地解析器与 Docling 服务之间的选择和降级。"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

from knowledge.ingestion.contracts import OcrPolicy, ParseRequest, ParseResult
from knowledge.ingestion.ports import DocumentParserPort


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfProfile:
    pages: int
    bytes_per_page: float
    text_chars_per_page: float
    image_area_ratio: float = 0.0
    table_like_pages_ratio: float = 0.0
    multi_column_pages_ratio: float = 0.0


@dataclass(frozen=True)
class ParseQuality:
    """本地解析结果的轻量质量评估。"""

    acceptable: bool
    page_coverage: float
    chars_per_page: float
    replacement_ratio: float
    traceable_ratio: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PdfRouteDecision:
    """PDF 文档级路由决策，可直接记录到建库诊断中。"""

    parser_name: Literal["local", "docling"]
    ocr_policy: OcrPolicy
    allow_fallback: bool
    reasons: tuple[str, ...] = ()


class PdfExcludedError(RuntimeError):
    """文档超出建库业务范围，应记录为排除而非解析失败。"""


_NUMERIC_TOKEN_RE = re.compile(r"(?<!\w)[-+]?[\d,.]+%?(?!\w)")


def _sample_page_indices(page_count: int, sample_pages: int) -> tuple[int, ...]:
    """均匀抽样首页、中间页和尾页，避免只看封面造成误判。"""
    sampled = min(max(sample_pages, 1), page_count)
    if sampled == page_count:
        return tuple(range(page_count))
    if sampled == 1:
        return (0,)
    return tuple(
        sorted(
            {
                round(index * (page_count - 1) / (sampled - 1))
                for index in range(sampled)
            }
        )
    )


def _is_table_like(text: str) -> bool:
    """用数字密集行识别财务表格倾向，不在预检查阶段运行重型表格模型。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numeric_lines = sum(
        1 for line in lines if len(_NUMERIC_TOKEN_RE.findall(line)) >= 3
    )
    return numeric_lines >= 3 and numeric_lines / max(len(lines), 1) >= 0.12


def _dict_block_text(block: dict[str, object]) -> str:
    """从 PyMuPDF dict 文字块中合并 span；该格式本身没有顶层 text 字段。"""
    parts: list[str] = []
    for line in block.get("lines") or []:  # type: ignore[union-attr]
        for span in line.get("spans") or []:
            parts.append(str(span.get("text") or ""))
    return "".join(parts).strip()


def _is_multi_column(blocks: list[dict[str, object]], page_width: float) -> bool:
    """根据文字块的横向分布判断明显双栏/多栏倾向。"""
    middle = page_width / 2
    text_blocks = [
        block
        for block in blocks
        if block.get("type") == 0 and len(_dict_block_text(block)) >= 20
    ]
    left = [
        block
        for block in text_blocks
        if float(block["bbox"][2]) <= middle * 1.1  # type: ignore[index]
    ]
    right = [
        block
        for block in text_blocks
        if float(block["bbox"][0]) >= middle * 0.9  # type: ignore[index]
    ]
    return len(left) >= 2 and len(right) >= 2


def inspect_pdf_profile(file_path: str, sample_pages: int = 5) -> PdfProfile:
    """低成本抽样 PDF，识别扫描、图像、表格和多栏倾向。"""
    import fitz

    path = Path(file_path)
    with fitz.open(path) as document:
        pages = len(document)
        indices = _sample_page_indices(pages, sample_pages)
        text_chars = 0
        image_area_ratios: list[float] = []
        table_like_pages = 0
        multi_column_pages = 0
        for index in indices:
            page = document[index]
            page_dict = page.get_text("dict")
            text = page.get_text().strip()
            text_chars += len(text)
            if _is_table_like(text):
                table_like_pages += 1
            blocks = list(page_dict.get("blocks") or [])
            if _is_multi_column(blocks, page.rect.width):
                multi_column_pages += 1
            page_area = max(page.rect.width * page.rect.height, 1.0)
            image_area = sum(
                max(0.0, float(block["bbox"][2]) - float(block["bbox"][0]))
                * max(0.0, float(block["bbox"][3]) - float(block["bbox"][1]))
                for block in blocks
                if block.get("type") == 1 and block.get("bbox")
            )
            image_area_ratios.append(min(image_area / page_area, 1.0))
    sampled = max(len(indices), 1)
    return PdfProfile(
        pages=pages,
        bytes_per_page=path.stat().st_size / max(pages, 1),
        text_chars_per_page=text_chars / max(sampled, 1),
        image_area_ratio=sum(image_area_ratios) / sampled,
        table_like_pages_ratio=table_like_pages / sampled,
        multi_column_pages_ratio=multi_column_pages / sampled,
    )


def assess_local_parse_quality(
    result: ParseResult,
    profile: PdfProfile,
    *,
    min_page_coverage: float = 0.65,
    min_chars_per_page: int = 80,
    max_replacement_ratio: float = 0.005,
    min_traceable_ratio: float = 0.95,
) -> ParseQuality:
    """判断本地解析是否足以直接建库。"""
    chunks = result.chunks
    content = "\n".join(chunk.content for chunk in chunks)
    valid_pages = {
        page
        for chunk in chunks
        for page in chunk.source_pages
        if 1 <= page <= profile.pages
    }
    page_coverage = len(valid_pages) / max(profile.pages, 1)
    chars_per_page = len(content) / max(profile.pages, 1)
    replacement_ratio = content.count("\ufffd") / max(len(content), 1)
    traceable_ratio = (
        sum(chunk.source_page is not None for chunk in chunks) / max(len(chunks), 1)
    )

    reasons: list[str] = []
    if not chunks:
        reasons.append("本地解析未产生内容")
    if profile.pages >= 3 and page_coverage < min_page_coverage:
        reasons.append(f"页码覆盖率过低（{page_coverage:.0%}）")
    if chars_per_page < min_chars_per_page:
        reasons.append(f"平均文本量过低（{chars_per_page:.0f} 字/页）")
    if replacement_ratio > max_replacement_ratio:
        reasons.append(f"乱码比例过高（{replacement_ratio:.2%}）")
    if chunks and traceable_ratio < min_traceable_ratio:
        reasons.append(f"可溯源片段比例过低（{traceable_ratio:.0%}）")
    return ParseQuality(
        acceptable=not reasons,
        page_coverage=page_coverage,
        chars_per_page=chars_per_page,
        replacement_ratio=replacement_ratio,
        traceable_ratio=traceable_ratio,
        reasons=tuple(reasons),
    )


class PdfParserRouter:
    """按显式模式或轻量 PDF 画像选择解析器。"""

    def __init__(
        self,
        *,
        local_parser: DocumentParserPort,
        docling_parser: DocumentParserPort,
        docling_enabled: bool,
        strict_mode: bool = False,
        mode: str = "auto",
        image_bytes_per_page: int = 307_200,
        min_text_chars_per_page: int = 100,
        min_image_area_ratio: float = 0.15,
        min_table_like_pages_ratio: float = 0.6,
        min_multi_column_pages_ratio: float = 0.4,
        max_pdf_pages: int = 200,
        local_min_page_coverage: float = 0.65,
        local_min_chars_per_page: int = 80,
        local_max_replacement_ratio: float = 0.005,
        local_min_traceable_ratio: float = 0.95,
        sample_pages: int = 5,
        profile_provider: Callable[[str, int], PdfProfile] = inspect_pdf_profile,
    ) -> None:
        if mode not in {"auto", "local", "docling"}:
            raise ValueError("解析模式必须是 auto、local 或 docling")
        self._local = local_parser
        self._docling = docling_parser
        self._docling_enabled = docling_enabled
        self._strict_mode = strict_mode
        self._mode = mode
        self._image_bytes_per_page = image_bytes_per_page
        self._min_text_chars_per_page = min_text_chars_per_page
        self._min_image_area_ratio = min_image_area_ratio
        self._min_table_like_pages_ratio = min_table_like_pages_ratio
        self._min_multi_column_pages_ratio = min_multi_column_pages_ratio
        self._max_pdf_pages = max_pdf_pages
        self._local_min_page_coverage = local_min_page_coverage
        self._local_min_chars_per_page = local_min_chars_per_page
        self._local_max_replacement_ratio = local_max_replacement_ratio
        self._local_min_traceable_ratio = local_min_traceable_ratio
        self._sample_pages = sample_pages
        self._profile_provider = profile_provider

    async def parse(self, request: ParseRequest) -> ParseResult:
        normalized = request.normalized()
        profile = await self._inspect_profile(normalized)
        if profile is not None and profile.pages > self._max_pdf_pages:
            raise PdfExcludedError(
                f"PDF 共 {profile.pages} 页，超过建库上限 {self._max_pdf_pages} 页"
            )
        decision = self._select(normalized, profile)
        if decision.parser_name == "local":
            result = await self._local.parse(normalized)
            if not self._docling_enabled or not decision.allow_fallback:
                return replace(result, route_reasons=decision.reasons)
            quality = self._assess_local_quality(result, profile)
            if quality.acceptable:
                return replace(result, route_reasons=decision.reasons)
            reason = "；".join(quality.reasons)
            logger.warning("本地解析质量未达标，转交 Docling：%s", reason)
            try:
                takeover_request = replace(
                    normalized,
                    ocr_policy=(
                        "disabled"
                        if normalized.ocr_policy == "disabled"
                        else "auto"
                    ),
                )
                takeover = await self._docling.parse(takeover_request)
                return replace(
                    takeover,
                    route_reasons=decision.reasons
                    + (f"本地解析质量未达标：{reason}",),
                )
            except Exception as exc:
                if self._strict_mode or not result.has_content:
                    raise
                logger.warning("Docling 接管失败，保留本地解析结果：%s", exc)
                return replace(
                    result,
                    warnings=result.warnings
                    + (f"本地解析质量告警：{reason}", f"Docling 接管失败：{exc}"),
                    route_reasons=decision.reasons,
                )

        try:
            docling_request = replace(normalized, ocr_policy=decision.ocr_policy)
            result = await self._docling.parse(docling_request)
            return replace(result, route_reasons=decision.reasons)
        except Exception as exc:
            if self._strict_mode or not decision.allow_fallback:
                raise
            logger.warning("Docling 解析失败，回退本地解析器：%s", exc)
            fallback = await self._local.parse(normalized)
            return replace(
                fallback,
                warnings=fallback.warnings + (f"Docling 回退原因：{exc}",),
                route_reasons=decision.reasons,
            )

    async def _inspect_profile(self, request: ParseRequest) -> PdfProfile | None:
        try:
            return await asyncio.to_thread(
                self._profile_provider,
                request.file_path,
                self._sample_pages,
            )
        except Exception as exc:
            logger.warning("PDF 画像失败，无法应用复杂版面路由：%s", exc)
            return None

    def _select(
        self,
        request: ParseRequest,
        profile: PdfProfile | None,
    ) -> PdfRouteDecision:
        hint = request.parser_hint
        if hint == "local":
            return PdfRouteDecision(
                parser_name="local",
                ocr_policy="disabled",
                allow_fallback=False,
                reasons=("请求显式指定本地解析器",),
            )
        if hint == "docling":
            if not self._docling_enabled:
                raise RuntimeError("Docling 解析器未启用")
            return PdfRouteDecision(
                parser_name="docling",
                ocr_policy=request.ocr_policy,
                allow_fallback=False,
                reasons=("请求显式指定 Docling",),
            )
        if self._mode == "local" or not self._docling_enabled:
            reason = "全局解析模式为 local" if self._mode == "local" else "Docling 未启用"
            return PdfRouteDecision("local", "disabled", False, (reason,))
        if self._mode == "docling":
            return PdfRouteDecision(
                "docling",
                request.ocr_policy,
                False,
                ("全局解析模式为 docling",),
            )

        if request.ocr_policy == "force":
            return PdfRouteDecision(
                "docling",
                "force",
                True,
                ("请求显式要求强制 OCR",),
            )

        if profile is None:
            return PdfRouteDecision(
                "local",
                "disabled",
                True,
                ("PDF 画像不可用，先尝试本地解析",),
            )

        reasons: list[str] = []
        if profile.bytes_per_page >= self._image_bytes_per_page:
            reasons.append("单页文件体积偏高")
        if profile.text_chars_per_page < self._min_text_chars_per_page:
            reasons.append("可提取文本密度过低")
        if profile.image_area_ratio >= self._min_image_area_ratio:
            reasons.append("图片覆盖面积偏高")
        if profile.table_like_pages_ratio >= self._min_table_like_pages_ratio:
            reasons.append("数字表格页面占比偏高")
        if profile.multi_column_pages_ratio >= self._min_multi_column_pages_ratio:
            reasons.append("多栏页面占比偏高")

        if not reasons:
            return PdfRouteDecision(
                "local",
                "disabled",
                True,
                ("文本层和版面适合快速解析",),
            )

        scanned = (
            profile.text_chars_per_page < self._min_text_chars_per_page
            and profile.image_area_ratio >= self._min_image_area_ratio
        )
        return PdfRouteDecision(
            "docling",
            (
                "disabled"
                if request.ocr_policy == "disabled"
                else "force"
                if scanned
                else "auto"
            ),
            True,
            tuple(reasons),
        )

    def _assess_local_quality(
        self,
        result: ParseResult,
        profile: PdfProfile | None,
    ) -> ParseQuality:
        if profile is None:
            return ParseQuality(
                acceptable=result.has_content,
                page_coverage=1.0 if result.has_content else 0.0,
                chars_per_page=float(len("".join(c.content for c in result.chunks))),
                replacement_ratio=0.0,
                traceable_ratio=1.0,
                reasons=() if result.has_content else ("本地解析未产生内容",),
            )
        return assess_local_parse_quality(
            result,
            profile,
            min_page_coverage=self._local_min_page_coverage,
            min_chars_per_page=self._local_min_chars_per_page,
            max_replacement_ratio=self._local_max_replacement_ratio,
            min_traceable_ratio=self._local_min_traceable_ratio,
        )
