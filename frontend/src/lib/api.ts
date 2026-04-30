import type {
  AgentRunResponse,
  AuthTokenResponse,
  NeedsMoreInfoResponse,
} from "../types";

const BASE = "/api";

function authHeaders(token: string) {
  return { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string
  ) {
    super(detail);
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore parse errors
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export async function login(email: string, password: string): Promise<AuthTokenResponse> {
  const res = await fetch(`${BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return handleResponse<AuthTokenResponse>(res);
}

export async function register(email: string, password: string): Promise<AuthTokenResponse> {
  const res = await fetch(`${BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return handleResponse<AuthTokenResponse>(res);
}

export async function queryAgent(
  query: string,
  token: string
): Promise<AgentRunResponse | NeedsMoreInfoResponse> {
  const res = await fetch(`${BASE}/agent/query`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ query }),
  });
  return handleResponse<AgentRunResponse | NeedsMoreInfoResponse>(res);
}

export async function fetchHistory(token: string): Promise<AgentRunResponse[]> {
  const res = await fetch(`${BASE}/agent/history`, {
    headers: authHeaders(token),
  });
  return handleResponse<AgentRunResponse[]>(res);
}
