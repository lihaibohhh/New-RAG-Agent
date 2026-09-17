"""生成问答对的确定性证据校验。"""

import re

from eval.dataset.models import EvidenceUnit, QAPair


def compact_text(value: str) -> str:
    return re.sub(r"[\s,，]", "", str(value or "")).casefold()


def answer_numbers(value: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", compact_text(value)))


def validate_qa_pair(pair: QAPair, unit: EvidenceUnit) -> tuple[bool, tuple[str, ...]]:
    """校验问题、逐字证据、多 Chunk 覆盖和答案数字来源。"""
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

    evidence_indices: set[int] = set()
    for evidence in pair.evidence:
        if not 1 <= evidence.chunk_index <= len(unit.chunks):
            reasons.append("invalid_evidence_chunk_index")
            continue
        quote = compact_text(evidence.quote)
        if len(quote) < 6:
            reasons.append("evidence_too_short")
            continue
        evidence_indices.add(evidence.chunk_index)
        corpus = compact_text(unit.chunks[evidence.chunk_index - 1].page_content)
        if quote not in corpus:
            reasons.append("evidence_quote_not_found")

    if unit.scope == "multi_chunk" and len(evidence_indices) < 2:
        reasons.append("multi_chunk_evidence_incomplete")
    corpus_text = compact_text("\n".join(chunk.page_content for chunk in unit.chunks))
    missing_numbers = sorted(
        number for number in answer_numbers(answer) if number not in corpus_text
    )
    if missing_numbers:
        reasons.append("unsupported_answer_numbers:" + ",".join(missing_numbers))
    normalized_reasons = tuple(dict.fromkeys(reasons))
    return not normalized_reasons, normalized_reasons


_compact_text = compact_text
_answer_numbers = answer_numbers
