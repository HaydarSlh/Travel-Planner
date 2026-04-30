import { useCallback, useRef } from "react";
import type { AgentRunResponse, DestinationMeta, ToolCallRecord } from "../types";

export type StreamEvent =
  | { type: "token"; text: string }
  | { type: "tool_call"; tool_name: string; input: Record<string, unknown>; output: Record<string, unknown>; duration_ms: number }
  | { type: "done"; run: AgentRunResponse }
  | { type: "needs_more_info"; message: string; missing_fields: string[] }
  | { type: "error"; message: string };

interface UseStreamOptions {
  onEvent: (event: StreamEvent) => void;
}

export function useStream({ onEvent }: UseStreamOptions) {
  const abortRef = useRef<AbortController | null>(null);

  const stream = useCallback(
    async (query: string, token: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const res = await fetch("/api/agent/query/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ query }),
        signal: controller.signal,
      });

      if (!res.ok) {
        let detail = res.statusText;
        try {
          const body = await res.json();
          detail = body.detail ?? detail;
        } catch {
          // ignore
        }
        onEvent({ type: "error", message: detail });
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        onEvent({ type: "error", message: "No response body" });
        return;
      }

      const decoder = new TextDecoder();
      let buf = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buf += decoder.decode(value, { stream: true });

          // SSE lines are separated by \n\n
          const parts = buf.split("\n\n");
          buf = parts.pop() ?? "";

          for (const part of parts) {
            const line = part.trim();
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6);
            let payload: Record<string, unknown>;
            try {
              payload = JSON.parse(jsonStr);
            } catch {
              continue;
            }

            const eventType = payload.type as string;

            if (eventType === "token") {
              onEvent({ type: "token", text: payload.text as string });
            } else if (eventType === "tool_call") {
              onEvent({
                type: "tool_call",
                tool_name: payload.tool_name as string,
                input: payload.input as Record<string, unknown>,
                output: payload.output as Record<string, unknown>,
                duration_ms: payload.duration_ms as number,
              });
            } else if (eventType === "done") {
              const run: AgentRunResponse = {
                run_id: payload.run_id as string,
                answer: payload.answer as string,
                styles_predicted: (payload.styles_predicted as string[]) ?? [],
                destination_metadata: (payload.destination_metadata as DestinationMeta[]) ?? [],
                tool_calls: (payload.tool_calls as ToolCallRecord[]) ?? [],
                token_usage: (payload.token_usage as Record<string, unknown>) ?? {},
                created_at: new Date().toISOString(),
              };
              onEvent({ type: "done", run });
            } else if (eventType === "needs_more_info") {
              onEvent({
                type: "needs_more_info",
                message: payload.message as string,
                missing_fields: (payload.missing_fields as string[]) ?? [],
              });
            } else if (eventType === "error") {
              onEvent({ type: "error", message: payload.message as string });
            }
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          onEvent({ type: "error", message: (err as Error).message });
        }
      } finally {
        reader.releaseLock();
      }
    },
    [onEvent]
  );

  const abort = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { stream, abort };
}
