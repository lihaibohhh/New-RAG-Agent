# 运行与部署指南

本文覆盖 Docker、本地进程、Knowledge Service、Docling 和 MCP 的启动方式。

## 前置条件

- Python 3.10+
- Conda 环境 `new_agent`
- Docker Desktop 与 Docker Compose（容器部署）
- 可选 PostgreSQL
- Redis
- 已准备的模型缓存和私有文档目录

安装依赖：

```powershell
conda run -n new_agent python -m pip install -e . --group dev
```

复制并填写环境变量：

```powershell
Copy-Item .env.example .env
```

## Docker Compose

```powershell
docker compose config
docker compose build knowledge-service agent-app
docker compose up -d redis knowledge-service agent-app
docker compose ps
```

Compose 分别启动 Streamlit Agent 和单 Worker Knowledge Service。后者是 Chroma、
BM25、Embedding、Reranker 和 Chunk Store 的唯一在线所有者。

启动容器前，应停止其他直接打开相同 Chroma 目录的进程或容器，避免 HNSW 多进程争用。

默认地址：

| 服务 | 地址 |
|---|---|
| Streamlit | `http://127.0.0.1:8501` |
| FastAPI | `http://127.0.0.1:8000` |
| Swagger | `http://127.0.0.1:8000/docs` |
| Knowledge Service | `http://127.0.0.1:8001` |

## 健康检查

```powershell
curl.exe http://127.0.0.1:8001/api/v1/health/live
curl.exe http://127.0.0.1:8001/api/v1/health/ready
```

`live` 只确认 HTTP 进程存活；`ready` 会检查真实知识库资源，因此能够发现 Chroma、
Chunk Store 或预热失败。

首次启动可能从现有 Chroma 生成 `data/knowledge/chunks.sqlite3` 并预热检索组件。
运行状态以 `/api/v1/runtime/warmup` 为准。

## 模型缓存

Knowledge Service 默认使用离线模型缓存：

```env
HF_HUB_OFFLINE=true
TRANSFORMERS_OFFLINE=true
```

首次部署且缓存为空时，可以临时设为 `false` 完成模型下载。确认缓存完整后恢复离线
模式，降低服务重启时受外部网络影响的概率。

如需指定 Python 包镜像：

```powershell
docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple knowledge-service agent-app
```

## PostgreSQL 与 Redis

Agent 在容器中而 PostgreSQL 位于 Windows 宿主机时，使用：

```env
POSTGRES_DOCKER_DB_URL=postgresql://user:password@host.docker.internal:5432/finance_agent
```

Redis 不保存 Agent 会话，也不承担向量检索。它用于语义缓存、BM25 缓存、API 限流
和每日 Token 预算。不要把 Redis 端口直接暴露到公网。

## Docling 离线解析

Docling 只参与解析和建库，不参与日常在线查询。

```env
DOCLING_ENABLED=true
DOCLING_PARSER_MODE=auto
DOCLING_BASE_URL=http://127.0.0.1:5001
```

启动服务：

```powershell
docker compose --profile ingestion up -d docling
docker compose --profile ingestion ps
curl.exe http://127.0.0.1:5001/ready
```

由 Knowledge Service 发起增量建库。路径必须相对于
`KNOWLEDGE_SERVICE_INGESTION_ROOT`：

```powershell
curl.exe -X POST http://127.0.0.1:8001/api/v1/ingestions `
  -H "Content-Type: application/json" `
  -H "X-Knowledge-Service-Key: <your-key>" `
  -d '{"relative_path":"."}'
```

`auto` 模式会根据扫描特征、图片比例、表格、多栏布局和本地解析质量选择解析器。
超过 `DOCLING_MAX_PDF_PAGES` 的文档会在解析前拒绝。

完成建库后可停止：

```powershell
docker compose --profile ingestion stop docling
```

## 本地进程

Streamlit：

```powershell
conda run -n new_agent python -m streamlit run scripts/streamlit_app.py
```

FastAPI：

```powershell
conda run -n new_agent python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

MCP stdio：

```powershell
conda run -n new_agent python mcp_rag_server.py
```

安装项目后也可以直接运行 `financial-rag-mcp`。

## MCP 客户端接入

Inspector：

```powershell
npx -y @modelcontextprotocol/inspector -- conda run -n new_agent python mcp_rag_server.py
```

注册到 Codex：

```powershell
codex mcp add financial-rag --env MCP_EXPOSE_ADMIN_TOOLS=0 -- financial-rag-mcp
```

默认只需调用 `query_financial_reports`。诊断场景可以设置：

```env
MCP_EXPOSE_ADMIN_TOOLS=true
```

然后使用知识库检查和预热状态工具。MCP stdout 只能写协议消息；通过
`MCP_LOG_PATH` 指定日志文件，写入失败时会降级到 stderr。

## 公网部署检查

- 使用 HTTPS 和反向代理。
- 设置随机长 API Key、Knowledge Service Key 和 BM25 HMAC Secret。
- 不公开 Redis、PostgreSQL、Chroma 或 Docling 端口。
- 限制 Knowledge Service 建库根目录。
- 监控 readiness、请求失败率、TTFT、Token 和成本。
- 使用单 Worker Knowledge Service，避免本地索引的多写者问题。
