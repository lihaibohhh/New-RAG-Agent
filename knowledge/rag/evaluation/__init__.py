"""知识库评测专用用例，不进入线上 Agent 查询路径。"""

from knowledge.rag.evaluation.service import EvaluationRetrievalService


__all__ = ["EvaluationRetrievalService"]
