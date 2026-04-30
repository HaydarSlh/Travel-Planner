import { useState } from "react";
import { ChevronDown, ChevronRight, Clock, Wrench } from "lucide-react";
import clsx from "clsx";
import type { ToolCallRecord } from "../../types";

const TOOL_COLORS: Record<string, string> = {
  rag_retrieve: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
  live_conditions: "text-sky-400 bg-sky-500/10 border-sky-500/20",
  classify_destination: "text-violet-400 bg-violet-500/10 border-violet-500/20",
};

const DEFAULT_COLOR = "text-slate-400 bg-slate-500/10 border-slate-500/20";

function ToolBadge({ name }: { name: string }) {
  const cls = TOOL_COLORS[name] ?? DEFAULT_COLOR;
  return (
    <span className={clsx("px-2 py-0.5 rounded-full text-xs font-medium border", cls)}>
      {name}
    </span>
  );
}

interface TraceRowProps {
  tc: ToolCallRecord;
}

function TraceRow({ tc }: TraceRowProps) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-[#1e2540] rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-3 px-3 py-2.5 bg-[#131929] hover:bg-[#1a2240] transition-colors text-left"
      >
        {open ? (
          <ChevronDown className="w-3.5 h-3.5 text-slate-500 shrink-0" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-slate-500 shrink-0" />
        )}
        <Wrench className="w-3.5 h-3.5 text-slate-500 shrink-0" />
        <ToolBadge name={tc.tool_name} />
        <span className="flex-1" />
        <span className="flex items-center gap-1 text-xs text-slate-500">
          <Clock className="w-3 h-3" />
          {tc.duration_ms}ms
        </span>
      </button>
      {open && (
        <div className="px-3 py-2.5 bg-[#0f1117] space-y-2 text-xs">
          <div>
            <p className="text-slate-500 uppercase tracking-wide mb-1 font-medium">Input</p>
            <pre className="text-slate-300 whitespace-pre-wrap break-all leading-relaxed">
              {JSON.stringify(tc.input, null, 2)}
            </pre>
          </div>
          <div>
            <p className="text-slate-500 uppercase tracking-wide mb-1 font-medium">Output</p>
            <pre className="text-slate-300 whitespace-pre-wrap break-all leading-relaxed">
              {JSON.stringify(tc.output, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

interface Props {
  toolCalls: ToolCallRecord[];
}

export default function ToolTrace({ toolCalls }: Props) {
  const [panelOpen, setPanelOpen] = useState(false);
  if (toolCalls.length === 0) return null;

  return (
    <div className="border-t border-[#1e2540] bg-[#0d1120]">
      <button
        onClick={() => setPanelOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-4 py-2.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
      >
        {panelOpen ? (
          <ChevronDown className="w-3.5 h-3.5" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5" />
        )}
        <Wrench className="w-3.5 h-3.5" />
        <span className="font-medium">Tool trace</span>
        <span className="ml-1 px-1.5 py-0.5 rounded-full bg-[#1a2240] text-indigo-400 text-xs">
          {toolCalls.length}
        </span>
      </button>
      {panelOpen && (
        <div className="px-4 pb-4 space-y-1.5 max-h-64 overflow-y-auto">
          {toolCalls.map((tc, i) => (
            <TraceRow key={i} tc={tc} />
          ))}
        </div>
      )}
    </div>
  );
}
