import React, { useState } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import {
  Globe,
  LogOut,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
} from "lucide-react";
import clsx from "clsx";
import type { AgentRunResponse } from "../../types";

interface Props {
  children: React.ReactNode;
  history: AgentRunResponse[];
  onSelectHistory: (run: AgentRunResponse) => void;
  onNewChat: () => void;
  activeRunId?: string | null;
}

export default function AppShell({
  children,
  history,
  onSelectHistory,
  onNewChat,
  activeRunId,
}: Props) {
  const { isAuthenticated, email, logout } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  if (!isAuthenticated) return <Navigate to="/login" replace />;

  function truncate(text: string, max = 38) {
    return text.length > max ? text.slice(0, max) + "…" : text;
  }

  function relativeTime(iso: string) {
    const diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  return (
    <div className="flex h-screen bg-[#0f1117] overflow-hidden">
      {/* Sidebar */}
      <aside
        className={clsx(
          "flex flex-col bg-[#131929] border-r border-[#1a2035] transition-all duration-300 shrink-0",
          sidebarOpen ? "w-[280px]" : "w-0 overflow-hidden"
        )}
      >
        {/* Sidebar header */}
        <div className="flex items-center gap-3 px-4 py-4 border-b border-[#1a2035]">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
            <Globe className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold text-white text-sm">Travel Planner</span>
        </div>

        {/* New chat button */}
        <div className="px-3 py-3">
          <button
            onClick={onNewChat}
            className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl
                       border border-dashed border-[#2a3555] text-slate-400
                       hover:border-indigo-500 hover:text-indigo-300 hover:bg-indigo-500/5
                       transition-all text-sm font-medium"
          >
            <Plus className="w-4 h-4" />
            New trip search
          </button>
        </div>

        {/* History list */}
        <div className="flex-1 overflow-y-auto px-2 pb-4">
          {history.length === 0 ? (
            <p className="text-xs text-slate-500 px-3 py-2">No previous searches yet.</p>
          ) : (
            <div className="space-y-0.5">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wide px-2 py-2">
                Recent
              </p>
              {history.map((run) => (
                <button
                  key={run.run_id}
                  onClick={() => onSelectHistory(run)}
                  className={clsx(
                    "w-full text-left px-3 py-2.5 rounded-xl transition-all group",
                    activeRunId === run.run_id
                      ? "bg-indigo-600/20 border border-indigo-500/30"
                      : "hover:bg-[#1a2240] border border-transparent"
                  )}
                >
                  <div className="flex items-start gap-2">
                    <MessageSquare
                      className={clsx(
                        "w-3.5 h-3.5 mt-0.5 shrink-0",
                        activeRunId === run.run_id ? "text-indigo-400" : "text-slate-500"
                      )}
                    />
                    <div className="min-w-0">
                      <p
                        className={clsx(
                          "text-sm leading-snug truncate",
                          activeRunId === run.run_id ? "text-indigo-200" : "text-slate-300"
                        )}
                      >
                        {truncate(run.query ?? run.answer.slice(0, 60))}
                      </p>
                      <p className="text-xs text-slate-500 mt-0.5">
                        {relativeTime(run.created_at)}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* User footer */}
        <div className="border-t border-[#1a2035] px-3 py-3">
          <div className="flex items-center gap-2.5 px-2 py-2 rounded-xl">
            <div className="w-7 h-7 rounded-full bg-indigo-600 flex items-center justify-center text-xs font-bold text-white shrink-0">
              {email?.[0]?.toUpperCase() ?? "U"}
            </div>
            <p className="text-sm text-slate-300 truncate flex-1">{email}</p>
            <button
              onClick={logout}
              title="Sign out"
              className="text-slate-500 hover:text-red-400 transition-colors p-1 rounded-lg hover:bg-red-500/10"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* Main area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Top bar */}
        <header className="flex items-center h-12 px-4 border-b border-[#1a2035] bg-[#0f1117] shrink-0">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="text-slate-400 hover:text-slate-200 transition-colors p-1.5 rounded-lg hover:bg-[#1a2240]"
          >
            {sidebarOpen ? (
              <PanelLeftClose className="w-4 h-4" />
            ) : (
              <PanelLeftOpen className="w-4 h-4" />
            )}
          </button>
        </header>

        <main className="flex-1 overflow-hidden">{children}</main>
      </div>
    </div>
  );
}
