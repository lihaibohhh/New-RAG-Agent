"""评测数据集生成使用的结构化模型。"""

from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class EvidenceQuote(BaseModel):
    """模型给出的可机械核验的原文证据。"""

    chunk_index: int = Field(description="证据所在片段编号，从 1 开始")
    quote: str = Field(description="从对应片段逐字摘录的短句，不得改写")


class QAPair(BaseModel):
    """单个问答对。"""

    question: str = Field(
        description="问题，必须包含具体的公司名称、行业或产品名，禁止使用模糊代词"
    )
    ground_truth: str = Field(
        description="答案必须直接来自研报片段，不得补充片段中未出现的数字或结论"
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
    """一次结构化生成返回的问答集合。"""

    pairs: list[QAPair] = Field(
        default_factory=list,
        description="生成的 QA 对；无实质内容时返回空列表",
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
    """一次结构化生成返回的无答案候选集合。"""

    cases: list[NoAnswerCase] = Field(default_factory=list)


@dataclass(frozen=True)
class EvidenceUnit:
    """一次生成调用使用的一个或多个相关 Chunk。"""

    chunks: tuple[Document, ...]
    scope: Literal["single_chunk", "multi_chunk"]
