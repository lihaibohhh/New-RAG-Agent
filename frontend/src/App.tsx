import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { AgentApiError, ResearchAgentClient } from "./api/agentClient";
import {
  ClockIcon,
  DatabaseIcon,
  KeyIcon,
  PlusIcon,
  SendIcon,
  StopIcon,
  ToolIcon,
  TrashIcon,
} from "./icons";
import { loadApiKey, loadSessions, saveApiKey, saveSessions } from "./storage";
import type {
  AgentEvent,
  ChatMessage,
  HealthResponse,
  LocalSession,
  ToolRun,
  UsageSummary,
} from "./types";

const EMPTY_USAGE: UsageSummary = {
  modelCalls: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  totalCost: 0,
  currency: "CNY",
  totalMs: null,
  ttftMs: null,
  unpricedModelCount: 0,
};

const EXAMPLE_PROMPTS = [
  "汽车行业2026年十大趋势是什么？",
  "分析半导体行业供需变化，并给出来源",
  "结合内部研报梳理新能源车产业链机会",
];

function makeSession(title = "新研究会话"): LocalSession {
  return { id: crypto.randomUUID(), title, updatedAt: Date.now() };
}

function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function numberValue(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function formatDuration(milliseconds: number | null): string {
  if (milliseconds === null) return "—";
  return milliseconds < 1000
    ? `${milliseconds} ms`
    : `${(milliseconds / 1000).toFixed(1)} 秒`;
}

function formatCost(value: number, unpriced: number): string {
  if (unpriced > 0 && value === 0) return "未计价";
  return `¥${value.toFixed(4)}`;
}

function App() {
  const initialSessions = useMemo(() => loadSessions(), []);
  const initialSession = useMemo(
    () => initialSessions[0] ?? makeSession(),
    [initialSessions],
  );

  const [sessions, setSessions] = useState<LocalSession[]>(() =>
    initialSessions.length > 0 ? initialSessions : [initialSession],
  );
  const [activeSessionId, setActiveSessionId] = useState(initialSession.id);
  const [apiKey, setApiKey] = useState(loadApiKey);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [tools, setTools] = useState<ToolRun[]>([]);
  const [usage, setUsage] = useState<UsageSummary>(EMPTY_USAGE);
  const [input, setInput] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const messageEndRef = useRef<HTMLDivElement | null>(null);
  const toolSequenceRef = useRef(0);

  const client = useMemo(
    () => new ResearchAgentClient({ apiKey }),
    [apiKey],
  );

  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  useEffect(() => {
    saveApiKey(apiKey);
  }, [apiKey]);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const controller = new AbortController();
    client
      .health(controller.signal)
      .then((value) => {
        setHealth(value);
        setHealthError(false);
      })
      .catch(() => setHealthError(true));
    return () => controller.abort();
  }, [client]);

  const loadHistory = useCallback(
    async (sessionId: string) => {
      setActiveSessionId(sessionId);
      setError(null);
      setTools([]);
      setUsage(EMPTY_USAGE);
      setIsLoadingHistory(true);
      try {
        const history = await client.history(sessionId);
        const visibleMessages = history.messages.filter(
          (message) =>
            (message.type === "human" || message.type === "ai") &&
            message.content.trim(),
        );
        setMessages(
          visibleMessages.map((message, index) => ({
            id: message.id ?? `${sessionId}-${index}`,
            role: message.type === "human" ? "user" : "assistant",
            content: message.content,
          })),
        );
      } catch (caught) {
        if (caught instanceof AgentApiError && caught.status === 404) {
          setMessages([]);
        } else {
          setError(
            caught instanceof Error ? caught.message : "读取会话历史失败",
          );
        }
      } finally {
        setIsLoadingHistory(false);
      }
    },
    [client],
  );

  useEffect(() => {
    void loadHistory(initialSession.id);
  }, [initialSession.id, loadHistory]);

  const createSession = useCallback(() => {
    abortRef.current?.abort();
    const session = makeSession();
    setSessions((current) => [session, ...current]);
    setActiveSessionId(session.id);
    setMessages([]);
    setTools([]);
    setUsage(EMPTY_USAGE);
    setError(null);
    setInput("");
  }, []);

  const deleteCurrentSession = useCallback(async () => {
    const current = sessions.find((session) => session.id === activeSessionId);
    if (!current || !window.confirm(`确定删除“${current.title}”吗？`)) {
      return;
    }

    try {
      await client.deleteSession(activeSessionId);
    } catch (caught) {
      if (!(caught instanceof AgentApiError && caught.status === 404)) {
        setError(caught instanceof Error ? caught.message : "删除会话失败");
        return;
      }
    }

    const remaining = sessions.filter((session) => session.id !== activeSessionId);
    const next = remaining[0] ?? makeSession();
    const nextSessions = remaining.length > 0 ? remaining : [next];
    setSessions(nextSessions);
    setActiveSessionId(next.id);
    setMessages([]);
    setTools([]);
    setUsage(EMPTY_USAGE);
    void loadHistory(next.id);
  }, [activeSessionId, client, loadHistory, sessions]);

  const updateAssistant = useCallback(
    (assistantId: string, updater: (message: ChatMessage) => ChatMessage) => {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? updater(message) : message,
        ),
      );
    },
    [],
  );

  const handleEvent = useCallback(
    (event: AgentEvent, assistantId: string) => {
      const data = event.data;

      if (event.type === "token") {
        const delta = stringValue(data.delta);
        updateAssistant(assistantId, (message) => ({
          ...message,
          content: message.content + delta,
        }));
      } else if (event.type === "answer_commit") {
        const content = stringValue(data.content);
        updateAssistant(assistantId, (message) => ({
          ...message,
          content,
          failed: false,
        }));
      } else if (event.type === "tool_call") {
        toolSequenceRef.current += 1;
        setTools((current) => [
          ...current,
          {
            id: `${stringValue(data.name, "tool")}-${toolSequenceRef.current}`,
            name: stringValue(data.name, "unknown_tool"),
            status: "running",
            args:
              typeof data.args === "object" && data.args !== null
                ? (data.args as Record<string, unknown>)
                : {},
          },
        ]);
      } else if (event.type === "tool_result") {
        const toolName = stringValue(data.name, "unknown_tool");
        const ok = data.ok !== false;
        setTools((current) => {
          const target = [...current]
            .reverse()
            .find((tool) => tool.name === toolName && tool.status === "running");
          if (!target) {
            return [
              ...current,
              {
                id: `${toolName}-${Date.now()}`,
                name: toolName,
                status: ok ? "success" : "failed",
                args: {},
                meta: stringValue(data.meta),
              },
            ];
          }
          return current.map((tool) =>
            tool.id === target.id
              ? {
                  ...tool,
                  status: ok ? "success" : "failed",
                  meta: stringValue(data.meta),
                }
              : tool,
          );
        });
      } else if (event.type === "usage") {
        setUsage((current) => ({
          ...current,
          modelCalls: current.modelCalls + 1,
          promptTokens:
            current.promptTokens + numberValue(data.prompt_tokens),
          completionTokens:
            current.completionTokens + numberValue(data.completion_tokens),
          totalCost: current.totalCost + numberValue(data.cost),
          currency: stringValue(data.currency, current.currency),
          unpricedModelCount:
            current.unpricedModelCount + (data.cost_status === "estimated" ? 0 : 1),
        }));
      } else if (event.type === "error") {
        const detail = stringValue(data.detail, "Agent执行失败");
        const requestId = stringValue(data.request_id);
        setError(requestId ? `${detail} · request_id=${requestId}` : detail);
        updateAssistant(assistantId, (message) => ({
          ...message,
          failed: true,
          pending: false,
          content: message.content || "本次请求未能完成。",
        }));
      } else if (event.type === "done") {
        setUsage((current) => ({
          ...current,
          totalTokens: numberValue(data.total_tokens),
          totalCost: numberValue(data.total_cost, current.totalCost),
          currency: stringValue(data.currency, current.currency),
          totalMs: numberValue(data.total_ms),
          ttftMs:
            data.ttft_ms === null ? null : numberValue(data.ttft_ms, 0),
          unpricedModelCount: numberValue(
            data.unpriced_model_count,
            current.unpricedModelCount,
          ),
        }));
        updateAssistant(assistantId, (message) => ({
          ...message,
          pending: false,
        }));
      }
    },
    [updateAssistant],
  );

  const sendMessage = useCallback(
    async (rawMessage?: string) => {
      const prompt = (rawMessage ?? input).trim();
      if (!prompt || isStreaming) return;

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: prompt,
      };
      const assistantId = crypto.randomUUID();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        pending: true,
      };

      setMessages((current) => [...current, userMessage, assistantMessage]);
      setTools([]);
      setUsage(EMPTY_USAGE);
      setError(null);
      setInput("");
      setIsStreaming(true);

      setSessions((current) => {
        const updated = current.map((session) =>
          session.id === activeSessionId
            ? {
                ...session,
                title:
                  session.title === "新研究会话"
                    ? prompt.slice(0, 24)
                    : session.title,
                updatedAt: Date.now(),
              }
            : session,
        );
        return updated.sort((a, b) => b.updatedAt - a.updatedAt);
      });

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await client.streamEvents(
          prompt,
          activeSessionId,
          (event) => handleEvent(event, assistantId),
          controller.signal,
        );
      } catch (caught) {
        if (caught instanceof DOMException && caught.name === "AbortError") {
          updateAssistant(assistantId, (message) => ({
            ...message,
            pending: false,
            content: message.content || "生成已停止。",
          }));
        } else {
          const detail =
            caught instanceof AgentApiError
              ? `${caught.message}${caught.requestId ? ` · request_id=${caught.requestId}` : ""}`
              : caught instanceof Error
                ? caught.message
                : "请求失败";
          setError(detail);
          updateAssistant(assistantId, (message) => ({
            ...message,
            pending: false,
            failed: true,
            content: message.content || "无法连接Agent API。",
          }));
        }
      } finally {
        updateAssistant(assistantId, (message) => ({ ...message, pending: false }));
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [activeSessionId, client, handleEvent, input, isStreaming, updateAssistant],
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">研</div>
          <div>
            <strong>研析</strong>
            <span>Research Agent</span>
          </div>
        </div>

        <button className="new-session" onClick={createSession} type="button">
          <PlusIcon />
          新建研究会话
        </button>

        <div className="sidebar-section-label">最近会话</div>
        <nav className="session-list" aria-label="最近会话">
          {sessions.map((session) => (
            <button
              className={`session-item ${session.id === activeSessionId ? "active" : ""}`}
              key={session.id}
              onClick={() => void loadHistory(session.id)}
              type="button"
            >
              <span className="session-title">{session.title}</span>
              <span className="session-time">
                {new Date(session.updatedAt).toLocaleDateString("zh-CN", {
                  month: "numeric",
                  day: "numeric",
                })}
              </span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <label className="api-key-field">
            <span><KeyIcon /> API Key</span>
            <input
              autoComplete="off"
              onChange={(event) => setApiKey(event.target.value)}
              placeholder="本地演示密钥"
              type="password"
              value={apiKey}
            />
          </label>
          <p>仅保存在当前浏览器标签页，不会写入前端源码。</p>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">EVIDENCE-BASED RESEARCH</p>
            <h1>
              {sessions.find((session) => session.id === activeSessionId)?.title ??
                "研究会话"}
            </h1>
          </div>
          <div className="topbar-actions">
            <div className={`health-pill ${healthError ? "offline" : ""}`}>
              <span className="status-dot" />
              {healthError
                ? "API不可用"
                : health
                  ? `${health.checkpoint_backend ?? "unknown"} · 已就绪`
                  : "正在检查服务"}
            </div>
            <button
              aria-label="删除当前会话"
              className="icon-button danger"
              onClick={() => void deleteCurrentSession()}
              type="button"
            >
              <TrashIcon />
            </button>
          </div>
        </header>

        <section className="conversation" aria-live="polite">
          {isLoadingHistory ? (
            <div className="center-state"><span className="spinner" />正在读取PostgreSQL会话状态</div>
          ) : messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-symbol">R</div>
              <p className="eyebrow">RESEARCH COPILOT</p>
              <h2>从一个值得研究的问题开始</h2>
              <p>
                Agent会按需检索内部研报、调用联网工具，并通过FastAPI将执行过程实时推送到这里。
              </p>
              <div className="prompt-grid">
                {EXAMPLE_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => void sendMessage(prompt)}
                    type="button"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  <div className="message-avatar">
                    {message.role === "user" ? "你" : message.role === "assistant" ? "研" : "系"}
                  </div>
                  <div className="message-body">
                    <div className="message-meta">
                      {message.role === "user"
                        ? "你"
                        : message.role === "assistant"
                          ? "Research Agent"
                          : message.role === "tool"
                            ? "工具记录"
                            : "系统"}
                      {message.pending && <span className="thinking">生成中</span>}
                    </div>
                    <div className={message.failed ? "message-error" : "message-content"}>
                      {message.role === "assistant" && message.content ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.content}
                        </ReactMarkdown>
                      ) : (
                        message.content || (message.pending ? "正在组织答案…" : "")
                      )}
                      {message.pending && message.content && <span className="cursor" />}
                    </div>
                  </div>
                </article>
              ))}
              <div ref={messageEndRef} />
            </div>
          )}
        </section>

        {error && (
          <div className="error-banner" role="alert">
            <strong>请求未完成</strong>
            <span>{error}</span>
            <button onClick={() => setError(null)} type="button">关闭</button>
          </div>
        )}

        <footer className="composer-wrap">
          <div className="composer">
            <textarea
              disabled={isStreaming}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
              placeholder="输入研究问题，Enter发送，Shift + Enter换行"
              rows={2}
              value={input}
            />
            {isStreaming ? (
              <button className="send-button stop" onClick={stopStreaming} type="button">
                <StopIcon />
                停止
              </button>
            ) : (
              <button
                className="send-button"
                disabled={!input.trim()}
                onClick={() => void sendMessage()}
                type="button"
              >
                <SendIcon />
                发送
              </button>
            )}
          </div>
          <p>回答由模型生成，请结合原始来源核验关键结论。</p>
        </footer>
      </main>

      <aside className="inspector">
        <section className="inspector-section">
          <div className="section-heading">
            <div><ToolIcon /><h2>本轮实时轨迹</h2></div>
            <span>{tools.length}</span>
          </div>
          {tools.length === 0 ? (
            <div className="empty-panel">
              <ToolIcon />
              <p>本轮发生工具调用时会实时显示在这里</p>
            </div>
          ) : (
            <div className="tool-list">
              {tools.map((tool) => (
                <details className="tool-card" key={tool.id} open={tool.status === "running"}>
                  <summary>
                    <span className={`tool-status ${tool.status}`} />
                    <span>{tool.name}</span>
                    <small>
                      {tool.status === "running"
                        ? "执行中"
                        : tool.status === "success"
                          ? "完成"
                          : "失败"}
                    </small>
                  </summary>
                  <pre>{JSON.stringify(tool.args, null, 2)}</pre>
                  {tool.meta && <p>{tool.meta}</p>}
                </details>
              ))}
            </div>
          )}
        </section>

        <section className="inspector-section metrics-section">
          <div className="section-heading">
            <div><ClockIcon /><h2>本轮观测</h2></div>
          </div>
          <div className="metrics-grid">
            <div><span>总耗时</span><strong>{formatDuration(usage.totalMs)}</strong></div>
            <div><span>首字延迟</span><strong>{formatDuration(usage.ttftMs)}</strong></div>
            <div><span>Token</span><strong>{usage.totalTokens.toLocaleString()}</strong></div>
            <div><span>模型调用</span><strong>{usage.modelCalls} 次</strong></div>
          </div>
          <div className="cost-row">
            <span>估算成本</span>
            <strong>{formatCost(usage.totalCost, usage.unpricedModelCount)}</strong>
          </div>
        </section>

        <section className="inspector-section architecture-card">
          <div className="section-heading">
            <div><DatabaseIcon /><h2>当前链路</h2></div>
          </div>
          <ol>
            <li><span>1</span>React发送HTTP请求</li>
            <li><span>2</span>FastAPI建立SSE流</li>
            <li><span>3</span>Agent执行模型与工具</li>
            <li><span>4</span>Checkpoint写入PostgreSQL</li>
          </ol>
        </section>
      </aside>
    </div>
  );
}

export default App;
