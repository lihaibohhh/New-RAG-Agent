from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import eval.run_eval as eval_runner
from eval.answerability_metrics import (
    aggregate_answerability_metrics,
    evaluate_answer_behavior,
)
from eval.audit_dataset import audit_records
from eval.run_eval import (
    _judge_answerability_sync,
    compute_summary,
    write_csv_detail,
    write_json_summary,
)
from eval.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_retrieval,
)


def test_chunk_id_metrics_use_ranked_gold_labels() -> None:
    record = {
        "question": "测试问题",
        "ground_truth": "测试答案",
        "gold_chunk_ids": ["gold-1", "gold-2"],
        "gold_sources": [{"source_file": "reports/a.pdf", "pages": [3]}],
    }
    retrieved = [
        {"chunk_id": "gold-1", "source_file": "reports/a.pdf", "source_page": 3},
        {"chunk_id": "noise", "source_file": "reports/b.pdf", "source_page": 1},
        {"chunk_id": "gold-2", "source_file": "reports/a.pdf", "source_page": 4},
    ]

    result = evaluate_retrieval(record, retrieved, top_k=3)

    assert result["retrieval_label_mode"] == "chunk_id"
    assert result["hit_at_k"] == 1.0
    assert result["precision_at_k"] == pytest.approx(2 / 3)
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 1.0
    expected_dcg = 1 + 1 / math.log2(4)
    ideal_dcg = 1 + 1 / math.log2(3)
    assert result["ndcg_at_k"] == pytest.approx(expected_dcg / ideal_dcg)
    assert result["source_page_recall"] == 1.0


def test_source_page_labels_fall_back_without_chunk_id() -> None:
    record = {
        "question": "旧数据问题",
        "ground_truth": "旧数据答案",
        "source_file": "行业/report.pdf",
        "page": 8,
    }
    retrieved = [
        {
            "chunk_id": "chunk-8",
            "source_file": "D:/data/行业/report.pdf",
            "source_page": 8,
        }
    ]

    result = evaluate_retrieval(record, retrieved, top_k=3)

    assert result["retrieval_label_mode"] == "source_page_fallback"
    assert result["retrieval_evaluable"] is True
    assert result["hit_at_k"] == 1.0
    assert result["precision_at_k"] == pytest.approx(1 / 3)
    assert result["recall_at_k"] == 1.0


def test_no_answer_and_missing_labels_are_reported_separately() -> None:
    no_answer = evaluate_retrieval(
        {"question": "库中不存在的问题", "answerable": False},
        [],
        top_k=3,
    )
    missing = evaluate_retrieval(
        {"question": "缺标签", "ground_truth": "答案"},
        [],
        top_k=3,
    )

    summary = aggregate_retrieval_metrics([no_answer, missing])

    assert no_answer["retrieval_abstained"] is True
    assert "no_answer_accuracy" not in no_answer
    assert summary["no_answer_count"] == 1
    assert summary["retrieval_abstention_rate"] == 1.0
    assert summary["retrieval_missing_label_count"] == 1
    assert summary["retrieval_evaluable_count"] == 0


def test_retrieval_failure_is_not_counted_as_successful_abstention() -> None:
    no_answer = {
        **evaluate_retrieval(
            {"question": "库中不存在的问题", "answerable": False},
            [],
            top_k=3,
        ),
        "retrieve_ok": False,
    }

    summary = aggregate_retrieval_metrics([no_answer])

    assert summary["retrieval_abstention_rate"] is None
    assert summary["no_answer_retrieval_error_count"] == 1


@pytest.mark.parametrize(
    ("answerable", "behavior", "outcome"),
    [
        (True, "supported_answer", "correct_answer"),
        (True, "abstention", "false_refusal"),
        (True, "unsupported_answer", "unsupported_answer"),
        (False, "abstention", "correct_abstention"),
        (False, "unsupported_answer", "hallucinated_answer"),
        (False, "supported_answer", "negative_label_conflict"),
    ],
)
def test_answer_behavior_maps_to_supervised_outcome(
    answerable: bool,
    behavior: str,
    outcome: str,
) -> None:
    result = evaluate_answer_behavior({"answerable": answerable}, behavior)

    assert result["answerability_evaluable"] is True
    assert result["answerability_outcome"] == outcome


def test_answerability_metrics_keep_refusal_and_hallucination_separate() -> None:
    records = [
        evaluate_answer_behavior({"answerable": True}, "supported_answer"),
        evaluate_answer_behavior({"answerable": True}, "abstention"),
        evaluate_answer_behavior({"answerable": False}, "abstention"),
        evaluate_answer_behavior({"answerable": False}, "unsupported_answer"),
    ]

    summary = aggregate_answerability_metrics(records)

    assert summary["answerability_accuracy"] == 0.5
    assert summary["abstention_accuracy"] == 0.5
    assert summary["hallucination_rate"] == 0.5
    assert summary["false_refusal_rate"] == 0.5
    assert summary["answerability_judge_coverage"] == 1.0


def test_answerability_metrics_report_partial_judge_coverage() -> None:
    records = [
        evaluate_answer_behavior({"answerable": True}, "supported_answer"),
        evaluate_answer_behavior({"answerable": True}, "judge_error"),
    ]

    summary = aggregate_answerability_metrics(records)

    assert summary["answerability_evaluable_count"] == 1
    assert summary["answerability_judge_coverage"] == 0.5


def test_empty_response_counts_as_failure_not_correct_abstention() -> None:
    result = evaluate_answer_behavior(
        {"answerable": False},
        "empty_response",
    )
    summary = aggregate_answerability_metrics(
        [{"answerable": False, **result}]
    )

    assert result["answerability_evaluable"] is True
    assert result["answerability_correct"] == 0.0
    assert summary["abstention_accuracy"] == 0.0
    assert summary["empty_response_count"] == 1


class _FakeAnswerabilityJudge:
    def with_structured_output(self, *_args, **_kwargs):
        return self

    def bind(self, **_kwargs):
        return self

    def invoke(self, _prompt):
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    case_id="positive",
                    behavior="supported_answer",
                    rationale="回答有上下文支持",
                ),
                SimpleNamespace(
                    case_id="negative",
                    behavior="abstention",
                    rationale="明确说明资料不足",
                ),
            ]
        )


class _OmittingAnswerabilityJudge(_FakeAnswerabilityJudge):
    def __init__(self) -> None:
        self.invoke_count = 0

    def invoke(self, _prompt):
        self.invoke_count += 1
        if self.invoke_count == 1:
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        case_id="positive",
                        behavior="supported_answer",
                        rationale="批次返回了正样本",
                    )
                ]
            )
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    case_id="negative",
                    behavior="abstention",
                    rationale="单条重试恢复",
                )
            ]
        )


def test_answerability_judge_compares_behavior_with_dataset_label() -> None:
    records = [
        {
            "case_id": "positive",
            "question": "收入是多少？",
            "contexts": ["收入为100亿元。"],
            "answer": "收入为100亿元。",
            "answerable": True,
        },
        {
            "case_id": "negative",
            "question": "未披露的利润是多少？",
            "contexts": ["材料只披露收入。"],
            "answer": "现有资料未提供利润，无法确定。",
            "answerable": False,
        },
    ]

    judged = _judge_answerability_sync(records, _FakeAnswerabilityJudge())

    assert judged[0]["answerability_outcome"] == "correct_answer"
    assert judged[1]["answerability_outcome"] == "correct_abstention"


def test_answerability_judge_retries_a_missing_case_individually() -> None:
    records = [
        {
            "case_id": "positive",
            "question": "收入是多少？",
            "contexts": ["收入为100亿元。"],
            "answer": "收入为100亿元。",
            "answerable": True,
        },
        {
            "case_id": "negative",
            "question": "未披露的利润是多少？",
            "contexts": ["材料只披露收入。"],
            "answer": "现有资料未提供利润，无法确定。",
            "answerable": False,
        },
    ]
    judge = _OmittingAnswerabilityJudge()

    judged = _judge_answerability_sync(records, judge)

    assert judge.invoke_count == 2
    assert judged[0]["answerability_outcome"] == "correct_answer"
    assert judged[1]["answerability_outcome"] == "correct_abstention"
    assert judged[1]["answerability_judge_rationale"] == "单条重试恢复"


def test_summary_separates_answerable_and_no_answer_groups() -> None:
    records = [
        {
            "answerable": True,
            **evaluate_answer_behavior({"answerable": True}, "supported_answer"),
        },
        {
            "answerable": False,
            **evaluate_answer_behavior({"answerable": False}, "abstention"),
        },
    ]

    summary = compute_summary(records)

    assert summary["by_answerability"]["answerable"]["n"] == 1
    assert summary["by_answerability"]["no_answer"]["n"] == 1
    assert summary["global"]["answerability_accuracy"] == 1.0


def test_summary_prefers_end_to_end_retrieval_latency() -> None:
    records = [
        {
            "retrieval_timings": {"total": 0.2, "evaluation_total": 2.5},
            "retrieve_ok": True,
        },
        {
            "retrieval_timings": {"total": 0.4},
            "retrieve_ok": True,
        },
    ]

    summary = compute_summary(records)

    assert summary["global"]["retrieval_latency_p50_ms"] == 400.0
    assert summary["global"]["retrieval_latency_p95_ms"] == 2500.0


def test_build_ragas_llm_uses_deepseek_openai_compatible_client(
    monkeypatch,
) -> None:
    captured = {}
    native_client = object()
    ragas_llm = object()

    def fake_async_openai(**kwargs):
        captured["client_kwargs"] = kwargs
        return native_client

    def fake_llm_factory(model, **kwargs):
        captured["model"] = model
        captured["factory_kwargs"] = kwargs
        return ragas_llm

    fake_settings = SimpleNamespace(
        secrets=SimpleNamespace(
            DEEPSEEK_API_KEY="test-deepseek-key",
            DEEPSEEK_BASE_URL="https://deepseek.test/v1",
        ),
        llm=SimpleNamespace(llm_timeout=60, llm_retries=2),
    )
    monkeypatch.setattr(eval_runner, "settings", fake_settings)
    monkeypatch.setattr(eval_runner, "AsyncOpenAI", fake_async_openai)
    monkeypatch.setattr(
        eval_runner,
        "ragas_llm_factory",
        fake_llm_factory,
        raising=False,
    )

    result = eval_runner._build_ragas_llm("deepseek/deepseek-chat")

    assert result is ragas_llm
    assert captured["client_kwargs"] == {
        "api_key": "test-deepseek-key",
        "base_url": "https://deepseek.test/v1",
        "timeout": 60,
        "max_retries": 2,
    }
    assert captured["model"] == "deepseek-chat"
    assert captured["factory_kwargs"] == {
        "provider": "openai",
        "client": native_client,
        "adapter": "instructor",
    }


@pytest.mark.asyncio
async def test_ragas_collections_only_score_answerable_records(monkeypatch) -> None:
    captured: dict[str, object] = {}
    calls: dict[str, list[dict]] = {
        "context_precision": [],
        "context_recall": [],
        "faithfulness": [],
    }
    evaluator_llm = object()

    class FakeMetric:
        def __init__(self, name: str, value: float) -> None:
            self.name = name
            self.value = value

        async def ascore(self, **kwargs):
            calls[self.name].append(kwargs)
            return SimpleNamespace(value=self.value)

    monkeypatch.setattr(eval_runner, "_RAGAS_OK", True)
    monkeypatch.setattr(
        eval_runner,
        "ContextPrecision",
        lambda **_kwargs: FakeMetric("context_precision", 1.0),
        raising=False,
    )
    monkeypatch.setattr(
        eval_runner,
        "ContextRecall",
        lambda **_kwargs: FakeMetric("context_recall", 0.75),
        raising=False,
    )
    monkeypatch.setattr(
        eval_runner,
        "Faithfulness",
        lambda **_kwargs: FakeMetric("faithfulness", 0.9),
        raising=False,
    )
    monkeypatch.setattr(
        eval_runner,
        "_build_ragas_llm",
        lambda model_ref: captured.setdefault("model_ref", model_ref)
        and evaluator_llm,
    )
    records = [
        {
            "question": "有答案问题",
            "contexts": ["证据"],
            "ground_truth": "参考答案",
            "answer": "模型答案",
            "answerable": True,
        },
        {
            "question": "无答案问题",
            "contexts": ["不相关内容"],
            "ground_truth": "",
            "answer": "资料不足",
            "answerable": False,
        },
    ]

    evaluated = await eval_runner._run_ragas_async(
        records,
        "deepseek/deepseek-chat",
        retrieval_only=False,
    )

    assert captured["model_ref"] == "deepseek/deepseek-chat"
    assert calls["context_precision"] == [
        {
            "user_input": "有答案问题",
            "reference": "参考答案",
            "retrieved_contexts": ["证据"],
        }
    ]
    assert calls["context_recall"] == [
        {
            "user_input": "有答案问题",
            "reference": "参考答案",
            "retrieved_contexts": ["证据"],
        }
    ]
    assert calls["faithfulness"] == [
        {
            "user_input": "有答案问题",
            "response": "模型答案",
            "retrieved_contexts": ["证据"],
        }
    ]
    assert evaluated[0]["context_precision"] == 1.0
    assert evaluated[0]["context_recall"] == 0.75
    assert evaluated[0]["faithfulness"] == 0.9
    assert evaluated[1]["context_precision"] is None
    assert evaluated[1]["context_recall"] is None
    assert evaluated[1]["faithfulness"] is None
    assert evaluated[1]["ragas_error"] is None


@pytest.mark.asyncio
async def test_ragas_metric_failure_is_recorded_without_stopping_batch(
    monkeypatch,
) -> None:
    class SuccessfulMetric:
        async def ascore(self, **_kwargs):
            return SimpleNamespace(value=0.8)

    class FailingMetric:
        async def ascore(self, **_kwargs):
            raise RuntimeError("synthetic metric failure")

    monkeypatch.setattr(eval_runner, "_RAGAS_OK", True)
    monkeypatch.setattr(eval_runner, "_build_ragas_llm", lambda _model_ref: object())
    monkeypatch.setattr(
        eval_runner,
        "ContextPrecision",
        lambda **_kwargs: FailingMetric(),
        raising=False,
    )
    monkeypatch.setattr(
        eval_runner,
        "ContextRecall",
        lambda **_kwargs: SuccessfulMetric(),
        raising=False,
    )
    monkeypatch.setattr(
        eval_runner,
        "Faithfulness",
        lambda **_kwargs: SuccessfulMetric(),
        raising=False,
    )

    evaluated = await eval_runner._run_ragas_async(
        [
            {
                "question": "测试问题",
                "contexts": ["测试证据"],
                "ground_truth": "参考答案",
                "answer": "模型答案",
                "answerable": True,
            }
        ],
        "deepseek/deepseek-chat",
        retrieval_only=False,
    )

    assert evaluated[0]["context_precision"] is None
    assert evaluated[0]["context_recall"] == 0.8
    assert evaluated[0]["faithfulness"] == 0.8
    assert evaluated[0]["ragas_error"] == "context_precision: RuntimeError"
    assert compute_summary(evaluated)["global"]["ragas_error_count"] == 1


def test_dataset_audit_requires_stable_ids_and_chunk_labels() -> None:
    report = audit_records(
        [
            {
                "question": "有来源页但没有 chunk id",
                "ground_truth": "答案",
                "source_file": "report.pdf",
                "page": 1,
            },
            {
                "case_id": "case-2",
                "question": "无答案问题",
                "answerable": False,
            },
        ]
    )

    assert report["total_records"] == 2
    assert report["source_page_fallback"] == 1
    assert report["no_answer_records"] == 1
    assert report["issues"]["missing_case_id"] == 1
    assert report["issues"]["source_page_fallback_only"] == 1


def test_eval_reports_support_external_output_directory(tmp_path) -> None:
    summary = write_json_summary({"global": {}}, "test", tmp_path)
    detail = write_csv_detail([], "test", tmp_path)

    assert summary == tmp_path / "eval_summary_test.json"
    assert detail == tmp_path / "eval_detail_test.csv"
    assert summary.exists()
    assert detail.exists()
    header = detail.read_text(encoding="utf-8-sig").splitlines()[0]
    assert "answerability_outcome" in header
    assert "hallucination" in header
    assert "ragas_error" in header
    assert "retrieval_trace" in header
    assert "retrieval_configuration" in header


def test_dataset_audit_requires_review_for_schema_v3_candidates() -> None:
    report = audit_records(
        [
            {
                "schema_version": 3,
                "case_id": "case-3",
                "question": "中芯国际收入是多少？",
                "ground_truth": "收入为100亿元。",
                "answerable": True,
                "gold_chunk_ids": ["chunk-1"],
                "gold_evidence": [
                    {"chunk_id": "chunk-1", "quote": "收入为100亿元"}
                ],
                "evidence_validation": {"status": "passed"},
                "review_status": "pending",
            }
        ]
    )

    assert report["evidence_validated"] == 1
    assert report["review_statuses"] == {"pending": 1}
    assert report["issues"]["review_status_pending"] == 1
