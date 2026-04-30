import { useCallback, useEffect, useState } from "react";
import AppShell from "../components/Layout/AppShell";
import ChatInterface from "../components/Chat/ChatInterface";
import { useAuth } from "../context/AuthContext";
import { fetchHistory } from "../lib/api";
import type { AgentRunResponse } from "../types";

export default function PlannerPage() {
  const { token } = useAuth();
  const [history, setHistory] = useState<AgentRunResponse[]>([]);
  const [selectedRun, setSelectedRun] = useState<AgentRunResponse | null>(null);
  const [chatKey, setChatKey] = useState(0); // increment to reset ChatInterface

  useEffect(() => {
    if (!token) return;
    fetchHistory(token)
      .then(setHistory)
      .catch(() => {/* ignore — non-critical */});
  }, [token]);

  const handleRunComplete = useCallback((run: AgentRunResponse) => {
    setHistory((prev) => {
      // Deduplicate by run_id, newest first
      const filtered = prev.filter((r) => r.run_id !== run.run_id);
      return [run, ...filtered];
    });
  }, []);

  const handleSelectHistory = useCallback((run: AgentRunResponse) => {
    setSelectedRun(run);
    setChatKey((k) => k + 1);
  }, []);

  const handleNewChat = useCallback(() => {
    setSelectedRun(null);
    setChatKey((k) => k + 1);
  }, []);

  return (
    <AppShell
      history={history}
      onSelectHistory={handleSelectHistory}
      onNewChat={handleNewChat}
      activeRunId={selectedRun?.run_id ?? null}
    >
      <ChatInterface
        key={chatKey}
        initialRun={selectedRun}
        onRunComplete={handleRunComplete}
      />
    </AppShell>
  );
}
