import type { LocalSession } from "./types";

const SESSION_INDEX_KEY = "research-agent.sessions.v1";
const API_KEY_SESSION_KEY = "research-agent.api-key";

export function loadSessions(): LocalSession[] {
  try {
    const parsed: unknown = JSON.parse(
      window.localStorage.getItem(SESSION_INDEX_KEY) ?? "[]",
    );
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter(
        (item): item is LocalSession =>
          typeof item === "object" &&
          item !== null &&
          typeof (item as LocalSession).id === "string" &&
          typeof (item as LocalSession).title === "string" &&
          typeof (item as LocalSession).updatedAt === "number",
      )
      .sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    return [];
  }
}

export function saveSessions(sessions: LocalSession[]): void {
  window.localStorage.setItem(
    SESSION_INDEX_KEY,
    JSON.stringify(sessions.slice(0, 30)),
  );
}

export function loadApiKey(): string {
  return window.sessionStorage.getItem(API_KEY_SESSION_KEY) ?? "";
}

export function saveApiKey(apiKey: string): void {
  window.sessionStorage.setItem(API_KEY_SESSION_KEY, apiKey);
}
