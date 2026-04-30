import React, { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUp, Globe2 } from "lucide-react";
import clsx from "clsx";
import { useAuth } from "../../context/AuthContext";
import { useStream } from "../../hooks/useStream";
import type { AgentRunResponse, ChatMessage, ToolCallRecord } from "../../types";
import MessageBubble from "./MessageBubble";
import ToolTrace from "./ToolTrace";

let msgIdCounter = 0;
function nextId() { return String(++msgIdCounter); }

interface Props {
  initialRun?: AgentRunResponse | null;
  onRunComplete?: (run: AgentRunResponse) => void;
}

const SUGGESTIONS = [
  "Two weeks in July, $2,000 budget, warm beach not touristy",
  "Family trip to Europe in spring, culture + history, mid-range budget",
  "Solo adventure, cold mountains, hiking, under $1,500",
  "Romantic getaway, luxury resort, any warm destination",
];

export default function ChatInterface({ initialRun, onRunComplete }: Props) {
  const { token } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  // live tool calls accumulating during current stream
  const [streamingToolCalls, setStreamingToolCalls] = useState<ToolCallRecord[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // keep a ref to the current loading message id so the event handler closure can access it
  const loadingIdRef = useRef<string | null>(null);

  const { stream } = useStream({
    onEvent: useCallback((event) => {
      const loadingId = loadingIdRef.current;
      if (!loadingId) return;

      if (event.type === "token") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingId && m.role === "assistant"
              ? { ...m, text: m.text + event.text, loading: false }
              : m
          )
        );
      } else if (event.type === "tool_call") {
        setStreamingToolCalls((prev) => [
          ...prev,
          {
            tool_name: event.tool_name,
            input: event.input,
            output: event.output,
            duration_ms: event.duration_ms,
          },
        ]);
      } else if (event.type === "done") {
        const run = event.run;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingId
              ? { role: "assistant", run, text: run.answer, loading: false, id: loadingId }
              : m
          )
        );
        onRunComplete?.(run);
        setBusy(false);
        loadingIdRef.current = null;
      } else if (event.type === "needs_more_info") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingId
              ? { role: "clarify", text: event.message, id: loadingId }
              : m
          )
        );
        setBusy(false);
        loadingIdRef.current = null;
      } else if (event.type === "error") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingId
              ? { role: "clarify", text: `Error: ${event.message}`, id: loadingId }
              : m
          )
        );
        setBusy(false);
        loadingIdRef.current = null;
      }
    }, [onRunComplete]),
  });

  // Load a history run when user selects it from sidebar
  useEffect(() => {
    if (!initialRun) {
      setMessages([]);
      return;
    }
    const userMsg: ChatMessage = {
      role: "user",
      text: initialRun.query ?? "Previous search",
      id: nextId(),
    };
    const assistantMsg: ChatMessage = {
      role: "assistant",
      run: initialRun,
      text: initialRun.answer,
      loading: false,
      id: nextId(),
    };
    setMessages([userMsg, assistantMsg]);
  }, [initialRun]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || busy || !token) return;
      setInput("");
      setStreamingToolCalls([]);

      const userMsg: ChatMessage = { role: "user", text: text.trim(), id: nextId() };
      const loadingId = nextId();
      const loadingMsg: ChatMessage = {
        role: "assistant",
        run: null,
        text: "",
        loading: true,
        id: loadingId,
      };
      loadingIdRef.current = loadingId;
      setMessages((prev) => [...prev, userMsg, loadingMsg]);
      setBusy(true);

      try {
        await stream(text.trim(), token);
      } catch (err) {
        const errorText = err instanceof Error ? err.message : "An unexpected error occurred.";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingId
              ? { role: "clarify", text: `Error: ${errorText}`, id: loadingId }
              : m
          )
        );
        setBusy(false);
        loadingIdRef.current = null;
      }
    },
    [busy, token, stream]
  );

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }

  const isEmpty = messages.length === 0;

  // Show streaming tool calls while running; fall back to last completed run's tool calls
  const lastAssistant = [...messages].reverse().find(
    (m): m is Extract<ChatMessage, { role: "assistant" }> => m.role === "assistant" && !m.loading
  );
  const toolCalls = busy
    ? streamingToolCalls
    : (lastAssistant?.run?.tool_calls ?? []);

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-5">
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
            <div className="w-16 h-16 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center">
              <Globe2 className="w-8 h-8 text-indigo-400" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white mb-1.5">
                Where do you want to go?
              </h2>
              <p className="text-slate-400 text-sm max-w-sm">
                Describe your dream trip and I'll find the perfect destination — with weather, costs, and things to do.
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2 max-w-lg">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => sendMessage(s)}
                  className="px-3.5 py-2 rounded-xl bg-[#161b2e] border border-[#1e2540]
                             text-sm text-slate-300 hover:border-indigo-500/40 hover:text-indigo-300
                             hover:bg-indigo-500/5 transition-all text-left"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
        )}
        <div ref={bottomRef} />
      </div>

      {/* Tool trace panel — updates live as tool_call events arrive */}
      <ToolTrace toolCalls={toolCalls} />

      {/* Input area */}
      <div className="shrink-0 px-4 pb-4 pt-3 border-t border-[#1a2035] bg-[#0f1117]">
        <div
          className={clsx(
            "flex items-end gap-2 bg-[#161b2e] border rounded-2xl px-4 py-3 transition-all",
            busy ? "border-indigo-500/30" : "border-[#1e2540] focus-within:border-indigo-500/50"
          )}
        >
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            disabled={busy}
            placeholder="Describe your dream trip…"
            className="flex-1 bg-transparent text-slate-100 placeholder-slate-500 text-sm
                       resize-none outline-none leading-relaxed min-h-[24px] max-h-[160px]"
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={busy || !input.trim()}
            className={clsx(
              "w-8 h-8 rounded-xl flex items-center justify-center transition-all shrink-0",
              busy || !input.trim()
                ? "bg-[#1e2540] text-slate-600 cursor-not-allowed"
                : "bg-indigo-600 text-white hover:bg-indigo-500 shadow-md shadow-indigo-500/20"
            )}
          >
            <ArrowUp className="w-4 h-4" />
          </button>
        </div>
        <p className="text-xs text-slate-600 text-center mt-2">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
