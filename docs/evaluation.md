# 测试与评测指南

项目区分离线回归、确定性检索评测和调用真实模型的 RAGAS/端到端评测。

## 离线回归

优先运行与改动相关的最小测试：

```powershell
conda run -n new_agent python -m pytest tests/api/test_stream.py -q
conda run -n new_agent python -m pytest tests/test_skills.py -q
```

完整 API 回归：

```powershell
conda run -n new_agent python -m pytest tests/api -q
```

API 测试使用 FakeAgent 和 fakeredis，不访问真实 LLM、Redis、PostgreSQL 或外网。
`tests/test_agent.py` 是 Streamlit 入口，不应作为普通 pytest 全量测试导入。

## 评测模块

| 模块 | 职责 |
|---|---|
| `eval/dataset/` | Chunk 数据源、分层抽样、证据校验、问答生成和数据集 Bundle |
| `eval/pipeline/` | 检索、回答生成、行为裁判、RAGAS、聚合与报告 |
| `eval.run_eval` | CLI、配置解析和阶段编排 |

## 数据集生成与审核

生成候选集会调用外部模型并产生费用：

```powershell
conda run -n new_agent python -m eval.dataset `
  --output eval/results/eval_dataset_docling_v1.candidate.jsonl `
  --max_chunks 150 --n_per_chunk 1 `
  --multi_chunk_ratio 0.25 --no_answer_count 20
```

审核候选集后，将通过项标记为 `review_status=approved`，再执行严格审计：

```powershell
conda run -n new_agent python -m eval.dataset.audit `
  --dataset eval/dataset/eval_dataset_docling_v1.jsonl --strict
```

可回答样本会检查引文、答案数字和多 Chunk 覆盖。无答案样本仍需执行全库检索筛查并
人工确认，不能仅依赖自动生成标签。

## 确定性检索评测

不调用生成模型：

```powershell
conda run -n new_agent python -m eval.run_eval --deterministic_only --n 150
```

BM25、Vector、Hybrid 三路消融：

```powershell
conda run -n new_agent python -m eval.run_eval --deterministic_only --retrieval_mode bm25 --n 150
conda run -n new_agent python -m eval.run_eval --deterministic_only --retrieval_mode vector --n 150
conda run -n new_agent python -m eval.run_eval --deterministic_only --retrieval_mode hybrid --n 150
```

主要指标：

- Hit@K
- Precision@K
- Recall@K
- MRR
- nDCG@K
- P50/P95 检索延迟
- retrieval abstention rate

默认绕过语义查询缓存，并使用专用评测管道记录 BM25、Vector、Fusion、Filtered、
Reranker Input、Reranked 和 Final 阶段轨迹。

## RAGAS 与端到端评测

仅检索 RAGAS：

```powershell
conda run -n new_agent python -m eval.run_eval --retrieval_only --n 150
```

端到端回答评测：

```powershell
conda run -n new_agent python -m eval.run_eval --n 150
```

这些模式会加载生成或裁判模型，可能访问外部服务并产生费用。正式评测不建议关闭
回答行为裁判。

端到端指标包括：

| 指标 | 含义 |
|---|---|
| `abstention_accuracy` | 无答案问题中正确拒答的比例 |
| `hallucination_rate` | 无答案问题中仍给出无依据答案的比例 |
| `false_refusal_rate` | 可回答问题中错误拒答的比例 |
| `negative_label_conflict_rate` | 检索证据与无答案标签发生冲突的比例 |
| `answerability_accuracy` | 可回答/不可回答行为的综合准确率 |
| `answerability_judge_coverage` | 成功完成行为判定的样本比例 |

无答案样本不参与 Recall、MRR、nDCG 或 RAGAS 正样本均值。

## 容器化正式基线

正式基线推荐使用一次性评测容器。它只通过 Knowledge Service API 读取 Chunk，不挂载
Chroma、模型缓存或私有研报。

```powershell
$compose = @('-f', 'docker-compose.yml', '-f', 'docker-compose.dev.yml')

docker compose @compose --profile eval run --rm eval-runner `
  python -m eval.run_eval `
  --dataset /app/eval/dataset/eval_dataset_docling_v1.jsonl `
  --deterministic_only --retrieval_mode hybrid --n 150
```

输入数据集只读挂载，报告写入 `eval/results/`。正式 A/B 应优先使用
`gold_chunk_ids`；解析器或知识库版本变化后，不应继续把旧报告当作当前基线。

## 已记录的建库性能

| 指标 | 串行版 | 多进程版 | 变化 |
|---|---:|---:|---:|
| 全量建库总耗时 | ~57 min | ~28 min | 约 2 倍提速 |
| T1+T2 解析 | ~56 min | ~27.6 min | 4 进程并行 |
| T3 向量化 | 45.7s | 62.47s | 批次结构变化，略增 |
| T4 写入 | 24.4s | 49.48s | `add → upsert` 换取幂等重跑 |
| 崩溃重跑 | 易出现 DuplicateIDError | upsert 幂等 | 稳定性提升 |

这些数字只代表当时的数据集和机器环境，新的性能结论应重新测量并记录环境。
