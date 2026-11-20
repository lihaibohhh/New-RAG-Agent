export type AgentEventType =
  | "token"
  | "answer_commit"
  | "tool_call"
  | "tool_result"
  | "usage"
  | "done"
  | "error"
  | string;

export interface AgentEvent {
  type: AgentEventType;
  data: Record<string, unknown>;
}

export type MessageRole = "user" | "assistant" | "tool" | "system";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  pending?: boolean;
  failed?: boolean;
}

export interface ToolRun {
  id: string;
  name: string;
  status: "running" | "success" | "failed";
  args: Record<string, unknown>;
  meta?: string;
}

export interface UsageSummary {
  modelCalls: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  totalCost: number;
  currency: string;
  totalMs: number | null;
  ttftMs: number | null;
  unpricedModelCount: number;
}

export interface HealthResponse {
  status: string;
  agent_initialized: boolean;
  checkpoint_backend: string | null;
}

export interface HistoryMessage {
  type: string;
  content: string;
  id: string | null;
}

export interface HistoryResponse {
  session_id: string;
  messages: HistoryMessage[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface LocalSession {
  id: string;
  title: string;
  updatedAt: number;
}
