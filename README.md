# ReAct-RAG-Agent

> 面向金融与行业研究的可信证据型 ReAct-RAG Agent。

用户可以用自然语言完成：

```text
研究拆解 → 私有研报检索 → 公开信息补充 → 口径与来源校验 → 报告交付
```

项目支持 Markdown、Word 和 Excel 输出，并尽量保留来源文件、页码与 chunk 标识，
方便研究人员复核关键结论。

> **能力边界**：本项目是研究辅助系统，不替代分析师、会计师、律师或投资决策者。
> 模型生成的内容应在对外发布或用于重要决策前进行人工复核，不构成投资建议。

## 核心能力

| 能力 | 说明 |
|---|---|
| 可信研究工作流 | 内置行业/市场研究 Skill，约束来源、时效、统计口径、预测版本和最终质检 |
| 私有知识库检索 | BM25 + 向量双路召回、RRF 融合、Cross-Encoder 精排 |
| 来源追溯 | 尽量保留 `source_file`、`source_page` 和 `chunk_id` |
| 公开信息补充 | 通过 Web 搜索补充公告、政策和其他时效性信息 |
| 多格式交付 | 生成 Markdown、Word、Excel 等研究产物 |
| 上下文管理 | 按完整轮次裁剪历史，维护证据索引并支持可选历史压缩 |
| 会话持久化 | 支持 PostgreSQL、SQLite 和 Memory Checkpointer |
| 服务接入 | 提供 Streamlit UI、FastAPI REST/SSE 和 MCP stdio |
| 安全与观测 | API Key、限流、每日 Token 预算、Prometheus 指标和成本统计 |
| 离线评测 | FakeAgent/fakeredis 回归测试，以及 RAGAS 与检索指标管道 |

## 典型场景

- 从大量研报中定位公司、行业、指标或政策信息，并返回来源页码。
- 在同一请求中完成研究拆解、证据检索、公开信息补充和报告生成。
- 查询 SQLite 财务指标库，并保留数据来源信息。
- 将已有分析继续整理成 Word 报告或公司对比 Excel。
- 通过 REST、SSE 或 MCP 将研究能力接入其他客户端和工作流。

推荐在单轮请求中明确研究边界，例如：

```text
请研究全球半导体设备市场：
1. 数据截止到 2026-09-26；
2. 分析市场规模、竞争格局、区域结构和三年趋势；
3. 优先使用私有研报与公司/机构一手来源；
4. 区分历史事实、机构预测和你的分析判断；
5. 核对关键数值、单位、年份与 CAGR；
6. 最终生成带来源的 Markdown 报告。
```

## 系统架构

![Architecture](img_1.png)

```mermaid
flowchart LR
    U[User] --> UI[Streamlit / FastAPI]
    UI --> A[LangGraph ReAct Agent]

    A --> SK[Research Skill]
    A --> RAG[RAG Tool]
    A --> WEB[Web Search]
    A --> OUT[Word / Excel / Markdown]

    RAG --> KS[Knowledge Service]
    KS --> RET[BM25 + Vector + RRF + Reranker]
    RET --> CH[(Chroma)]
    RET --> CS[(Chunk Store)]
    KS --> RC[(Redis Cache)]

    A --> CP[(Checkpoint Store)]
```

核心运行链路：

```text
用户请求
  → 选择并加载匹配的 Skill
  → LangGraph ReAct 循环
  → RAG / Web / 文档工具
  → 证据整理与失败恢复
  → 无工具 finalize_model
  → 返回回答、来源、工具轨迹和用量
```

架构边界、依赖方向和运行时策略见
[架构说明](docs/architecture.md) 与
[设计取舍](docs/design-decisions.md)。

## 快速开始

### 1. 安装

推荐 Python 3.10+，项目开发统一使用 Conda 环境 `new_agent`：

```powershell
git clone https://github.com/lihaibohhh/ReAct-RAG-Agent.git
cd ReAct-RAG-Agent/src
conda run -n new_agent python -m pip install -e . --group dev
```

### 2. 配置

复制示例配置：

```powershell
Copy-Item .env.example .env
```

至少配置模型密钥、Knowledge Service 和持久化后端。示例：

```env
MODEL=deepseek/deepseek-flash
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key

RAG_RUNTIME_MODE=remote
KNOWLEDGE_SERVICE_URL=http://127.0.0.1:8001
KNOWLEDGE_SERVICE_API_KEY=replace-with-a-random-secret

CHECKPOINT_BACKEND=postgres
POSTGRES_DB_URL=postgresql://user:password@127.0.0.1:5432/finance_agent
REDIS_URL=redis://127.0.0.1:6379/0
```

不要提交包含真实密钥的 `.env`。完整参数说明见
[配置指南](docs/configuration.md)。

### 3. Docker Compose 启动

```powershell
docker compose config
docker compose build knowledge-service agent-app
docker compose up -d redis knowledge-service agent-app
docker compose ps
```

服务地址：

- Streamlit：`http://localhost:8501`
- FastAPI Swagger：`http://localhost:8000/docs`
- Knowledge Service 就绪检查：`http://localhost:8001/api/v1/health/ready`

首次部署、离线模型缓存和 Docling 建库流程见
[部署指南](docs/deployment.md)。

### 4. 本地启动

```powershell
# Streamlit
conda run -n new_agent python -m streamlit run tests/test_agent.py

# FastAPI
conda run -n new_agent python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# MCP stdio
conda run -n new_agent python mcp_rag_server.py
```

## API 示例

非流式调用：

```bash
curl -X POST http://localhost:8000/api/v1/chat/invoke \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "message": "研究半导体行业供需变化，并给出来源",
    "session_id": "demo-session"
  }'
```

SSE 流式端点为 `POST /api/v1/chat/stream`，事件类型包括：

```text
token | tool_call | tool_result | usage | done | error
```

健康检查与指标：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/metrics
```

## 测试

普通回归测试不访问真实 LLM、Redis 或外网：

```powershell
conda run -n new_agent python -m pytest tests/api -q
conda run -n new_agent python -m pytest tests/test_skills.py -q
```

RAG 数据集生成、确定性检索评测和 RAGAS 流程见
[评测指南](docs/evaluation.md)。真实评测可能访问模型并产生费用。

## 目录概览

```text
src/
├── api/                    # FastAPI、鉴权、限流、指标和版本化路由
├── knowledge/
│   ├── rag/                # 在线检索、缓存、精排和评测
│   ├── ingestion/          # 文档解析和增量建库
│   ├── runtime/            # Knowledge 组合根与共享资源
│   ├── client/             # Knowledge Service HTTP 客户端
│   ├── transport/          # 中立 HTTP Schema 与 Codec
│   └── server/             # Knowledge Service ASGI 入口
├── react_agent/
│   ├── agent/              # Agent 状态、工作流、策略和上下文管理
│   ├── skills/             # 内置研究工作流
│   ├── tools/              # RAG、Web、文档和 SQL 工具
│   ├── conversations/      # 会话契约与 Checkpointer
│   └── runtime/            # 应用组合根与生命周期
├── mcp_service/            # MCP stdio 适配器
├── eval/                   # 数据集与 RAGAS 评测
├── tests/                  # 离线测试和 Streamlit 入口
└── scripts/                # 数据检查与调试脚本
```

## 详细文档

| 文档 | 内容 |
|---|---|
| [架构说明](docs/architecture.md) | Agent、Knowledge、RAG、会话与服务边界 |
| [配置指南](docs/configuration.md) | 模型、Token、工具预算、持久化和 Knowledge 配置 |
| [部署指南](docs/deployment.md) | Docker、健康检查、Docling、MCP 和本地运行 |
| [评测指南](docs/evaluation.md) | 离线回归、检索指标、RAGAS 和数据集审核 |
| [设计取舍](docs/design-decisions.md) | PDF 解析、检索、缓存、工具协议和上下文管理 |

## 当前边界

- 图表主导型 PDF 的财务数据抽取仍需要进一步增强多模态能力。
- 自动生成的评测问题和无答案样本必须经过人工审核。
- 本地默认配置面向开发与演示；公网部署需要额外配置 HTTPS、密钥轮换和网络隔离。
- 任何研究报告都需要对关键数字、时间、来源和结论进行人工复核。

## License

MIT License
