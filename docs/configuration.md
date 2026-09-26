# 项目配置指南

配置优先级为：

```text
Conda/系统环境变量 > .env > react_agent/configuration/config.yaml > 代码默认值
```

从示例文件开始：

```powershell
Copy-Item .env.example .env
```

不要提交真实密钥、数据库连接串或内部服务凭据。

## 模型配置

常用参数：

| 参数 | 作用 |
|---|---|
| `MODEL` | 模型标识，例如 `deepseek/deepseek-flash` |
| `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` | Provider 密钥 |
| `LLM_TEMPERATURE` | 生成温度 |
| `LLM_MAX_TOKENS` | 单次模型调用的最大输出 Token |
| `LLM_CONTEXT_WINDOW_TOKENS` | 模型实际上下文窗口，用于本地预算计算 |
| `LLM_TIMEOUT` | 模型请求超时 |
| `LLM_RETRIES` | Provider 调用重试次数 |

输出限制会按 Provider 映射：OpenAI 使用 `max_completion_tokens`，DeepSeek 兼容接口
使用 `max_tokens`，Anthropic 使用原生 `max_tokens`。

## 输入与历史 Token

| 参数 | 作用 |
|---|---|
| `MAX_HISTORY_TOKENS` | 为会话历史分配的估算 Token 子预算 |
| `MAX_INPUT_TOKENS` | 单次模型输入的本地估算总上限 |
| `CONTEXT_SAFETY_MARGIN_TOKENS` | 为估算误差预留的安全余量 |
| `ENABLE_HISTORY_COMPACTION` | 是否允许压缩较旧的完整对话轮次 |
| `HISTORY_COMPACTION_MAX_OUTPUT_TOKENS` | 历史摘要最大输出 |
| `HISTORY_COMPACTION_RETRY_NEW_TOKENS` | 摘要失败后再次尝试所需的新内容门槛 |

有效输入上限取以下较小值：

```text
MAX_INPUT_TOKENS
模型窗口 - LLM_MAX_TOKENS - CONTEXT_SAFETY_MARGIN_TOKENS
```

`MAX_HISTORY_TOKENS` 只限制历史消息，不包括系统提示词、工具 Schema、当前轮和证据。
当前轮超过历史子预算时仍会保留，最后由总输入预算判断能否调用模型。

历史压缩不会修改 Checkpoint 原文，只影响当前模型输入。摘要调用会计入模型调用次数、
Token 和成本。

## Agent 与工具预算

| 参数 | 作用 |
|---|---|
| `MAX_MODEL_ROUNDS` | 当前用户轮内常规模型调用上限 |
| `MAX_TOOL_BATCHES` | 当前用户轮内工具批次上限 |
| `MAX_TOOL_RETRIES` | 工具失败后的恢复重试上限 |
| `RAG_CALL_LIMIT` | 当前用户轮内被接受执行的 RAG 调用上限 |
| `CONSECUTIVE_FAILURE_THRESHOLD` | RAG 连续返回无有效内容后主动终止的阈值 |
| `RECURSION_LIMIT` | LangGraph 图步骤的最后安全熔断 |
| `MAX_TOOL_OUTPUT_CHARS` | 单次工具 observation 的硬截断字符数 |

RAG 配额规则：

- 成功、未命中和执行失败的 RAG 调用都会占用一次配额。
- 执行前因配额不足而被拒绝的调用不占用配额。
- 其他工具不占用 `RAG_CALL_LIMIT`，但会占用共享的工具批次预算。
- 同一模型批次中的多个 RAG 调用只会放行剩余配额。
- 连续无有效内容达到 `CONSECUTIVE_FAILURE_THRESHOLD` 后，Agent 会停止继续检索并收口。

`RECURSION_LIMIT` 必须足以容纳所配置的模型轮次和工具批次。启动时会按当前图拓扑校验：

```text
最低递归上限
= 1 个准备步骤
+ MAX_MODEL_ROUNDS
+ 3 × MAX_TOOL_BATCHES
+ 2 个闭合/总结步骤
+ 4 个安全余量
```

## 工具与语言

| 参数 | 作用 |
|---|---|
| `ENABLE_TOOLS` | Agent 工具总开关 |
| `ENABLE_WEB_SEARCH` | Web 搜索开关 |
| `TAVILY_API_KEY` | Tavily 搜索密钥 |
| `MAX_SEARCH_RESULTS` | 搜索结果上限 |
| `LANGUAGE` | 默认响应语言 |
| `TIMEZONE` | 系统时间和日期判断所用时区 |
| `OUTPUT_DIR` | 文件工具的输出目录 |

## 会话持久化

| 参数 | 作用 |
|---|---|
| `CHECKPOINT_BACKEND` | `postgres`、`sqlite` 或 `memory` |
| `CHECKPOINT_DB_PATH` | SQLite Checkpointer 路径 |
| `POSTGRES_DB_URL` | 本地 PostgreSQL 连接串 |
| `POSTGRES_DOCKER_DB_URL` | 容器访问宿主 PostgreSQL 的连接串 |
| `POSTGRES_POOL_MIN_SIZE` | 最小连接池大小 |
| `POSTGRES_POOL_MAX_SIZE` | 最大连接池大小 |
| `POSTGRES_CONNECT_TIMEOUT_SECONDS` | 连接超时 |

Redis 只承担缓存、限流和预算：

```env
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_MAX_CONNECTIONS=20
```

## Knowledge Service

Agent、MCP 和在线评测推荐使用远程模式：

```env
RAG_RUNTIME_MODE=remote
KNOWLEDGE_SERVICE_URL=http://127.0.0.1:8001
KNOWLEDGE_SERVICE_API_KEY=replace-with-a-random-secret
KNOWLEDGE_SERVICE_REQUIRE_API_KEY=true
KNOWLEDGE_SERVICE_TIMEOUT=150
```

`RAG_RUNTIME_MODE=local` 只用于 Knowledge Service 本身和显式离线维护。远程模式缺少
URL 时应直接启动失败，不会静默打开本地 Chroma。

检索和存储参数包括：

- `RAG_DEVICE`
- `RAG_CPU_RERANK_CANDIDATES`
- `RAG_CUDA_RERANK_CANDIDATES`
- `RAG_RERANK_TOP_N`
- `SEMANTIC_CACHE_TTL`
- `SEMANTIC_CACHE_THRESHOLD`
- `CHROMA_DB_PATH`
- `CHUNK_STORE_PATH`

## Docling 与离线模型

```env
DOCLING_ENABLED=false
DOCLING_PARSER_MODE=auto
DOCLING_BASE_URL=http://127.0.0.1:5001
DOCLING_MAX_PDF_PAGES=200

HF_HUB_OFFLINE=true
TRANSFORMERS_OFFLINE=true
```

`auto` 模式只把扫描、图像密集、表格密集、多栏或本地解析质量不足的 PDF 转交
Docling。完整阈值以 [.env.example](../.env.example) 为准。
