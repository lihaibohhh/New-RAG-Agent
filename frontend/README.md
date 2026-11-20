# Research Agent Web

面向项目演示的轻量 React + TypeScript + Vite 前端。页面只负责交互和展示，Agent、会话持久化、鉴权、限流与用量统计均由 FastAPI 服务负责。

## 本地开发

先启动 `agent-api`，再运行：

```powershell
cd frontend
pnpm install
pnpm dev
```

打开 `http://127.0.0.1:5173`。Vite 会把 `/api/*` 代理到 `http://127.0.0.1:8010`。

## Docker

从仓库根目录运行：

```powershell
docker compose up -d --build agent-web
```

打开 `http://127.0.0.1:3000`。Nginx 提供静态页面，并将 `/api/*` 转发给 `agent-api:8010`。

## 请求链路

```text
React UI
  -> frontend/src/api/agentClient.ts
  -> POST /api/v1/chat/stream
  -> FastAPI
  -> AgentService / LangGraph
  -> LLM、RAG 与工具
  -> SSE token/answer_commit/tool_call/tool_result/usage/done
  -> React 增量更新页面
```

普通 `token` 事件按增量追加；`answer_commit` 是服务端校验后的完整最终回答，前端会覆盖当前助手正文，确保页面结果与最终会话 State 一致。

服务端暂不支持跨 thread 枚举会话，所以浏览器使用 `localStorage` 保存最近使用的 `session_id` 索引；消息历史仍通过 `/api/v1/sessions/{session_id}/history` 从服务端 Checkpointer 读取。

API Key 仅保存在 `sessionStorage`，适合本地演示，不应视作公开部署的用户认证方案。
