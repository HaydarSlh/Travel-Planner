export interface ToolCallRecord {
  tool_name: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  duration_ms: number;
}

export interface DestinationMeta {
  name: string;
  image_url: string | null;
  source_url: string | null;
}

export interface AgentRunResponse {
  run_id: string;
  answer: string;
  tool_calls: ToolCallRecord[];
  styles_predicted: string[];
  destination_metadata: DestinationMeta[];
  token_usage: Record<string, unknown>;
  created_at: string;
  /** Attached client-side — the original user query */
  query?: string;
}

export interface NeedsMoreInfoResponse {
  status: "needs_more_info";
  message: string;
  missing_fields: string[];
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
}

export interface UserInfo {
  id: string;
  email: string;
}

export type ChatMessage =
  | { role: "user"; text: string; id: string }
  | {
      role: "assistant";
      run: AgentRunResponse | null;
      text: string; // streamed so far (or full answer)
      loading: boolean;
      id: string;
    }
  | { role: "clarify"; text: string; id: string };
