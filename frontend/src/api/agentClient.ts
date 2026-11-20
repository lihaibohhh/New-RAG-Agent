import { decodeSse } from "./sse";
import type {
  AgentEvent,
  HealthResponse,
  HistoryResponse,
} from "../types";

export class AgentApiError extends Error {
  readonly status: number | null;
  readonly requestId: string | null;

  constructor(
    message: string,
    options: { status?: number | null; requestId?: string | null } = {},
  ) {
    super(message);
    this.name = "AgentApiError";
    this.status = options.status ?? null;
    this.requestId = options.requestId ?? null;
  }
}

interface ProblemDetails {
  detail?: unknown;
  title?: unknown;
  request_id?: unknown;
}

interface ClientOptions {
  baseUrl?: string;
  apiKey?: string;
}

export class ResearchAgentClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;

  constructor({ baseUrl = "/api/v1", apiKey = "" }: ClientOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey.trim();
  }

  private headers(accept = "application/json"): HeadersInit {
    const headers: Record<string, string> = { Accept: accept };
    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }
    return headers;
  }

  private async errorFromResponse(response: Response): Promise<AgentApiError> {
    let problem: ProblemDetails = {};
    try {
      problem = (await response.json()) as ProblemDetails;
    } catch {
      // Reverse proxies may return plain text or HTML instead of problem+json.
    }

    const message = String(
      problem.detail ?? problem.title ?? response.statusText ?? "API 请求失败",
    );
    const requestId = String(
      problem.request_id ?? response.headers.get("X-Request-Id") ?? "",
    ).trim();

    return new AgentApiError(message, {
      status: response.status,
      requestId: requestId || null,
    });
  }

  async health(signal?: AbortSignal): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health/ready`, { signal });
    if (!response.ok) {
      throw await this.errorFromResponse(response);
    }
    return (await response.json()) as HealthResponse;
  }

  async history(
    sessionId: string,
    signal?: AbortSignal,
  ): Promise<HistoryResponse> {
    const params = new URLSearchParams({ page: "1", page_size: "100" });
    const response = await fetch(
      `${this.baseUrl}/sessions/${encodeURIComponent(sessionId)}/history?${params}`,
      { headers: this.headers(), signal },
    );
    if (!response.ok) {
      throw await this.errorFromResponse(response);
    }
    return (await response.json()) as HistoryResponse;
  }

  async deleteSession(sessionId: string): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/sessions/${encodeURIComponent(sessionId)}`,
      { method: "DELETE", headers: this.headers() },
    );
    if (!response.ok) {
      throw await this.errorFromResponse(response);
    }
  }

  async streamEvents(
    message: string,
    sessionId: string,
    onEvent: (event: AgentEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const response = await fetch(`${this.baseUrl}/chat/stream`, {
      method: "POST",
      headers: {
        ...this.headers("text/event-stream"),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message, session_id: sessionId }),
      signal,
    });

    if (!response.ok) {
      throw await this.errorFromResponse(response);
    }
    if (!response.body) {
      throw new AgentApiError("浏览器没有提供可读取的流式响应体");
    }

    for await (const event of decodeSse(response.body)) {
      onEvent(event);
    }
  }
}
