import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Globe } from "lucide-react";
import clsx from "clsx";
import type { ChatMessage } from "../../types";
import DestinationCard from "./DestinationCard";

interface Props {
  message: ChatMessage;
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-1 py-1">
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
    </div>
  );
}

export default function MessageBubble({ message }: Props) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="max-w-[75%] px-4 py-3 rounded-2xl rounded-br-sm
                        bg-indigo-600 text-white text-sm leading-relaxed shadow-md">
          {message.text}
        </div>
      </div>
    );
  }

  if (message.role === "clarify") {
    return (
      <div className="flex gap-3 animate-fade-in">
        <div className="w-8 h-8 rounded-full bg-amber-500/20 border border-amber-500/30 flex items-center justify-center shrink-0 mt-1">
          <Globe className="w-4 h-4 text-amber-400" />
        </div>
        <div className="max-w-[80%] px-4 py-3 rounded-2xl rounded-tl-sm bg-[#1a2035] border border-amber-500/20 text-sm text-slate-300 leading-relaxed">
          {message.text}
        </div>
      </div>
    );
  }

  // assistant
  const run = message.run;
  const destinations = run?.destination_metadata ?? [];
  const styles = run?.styles_predicted ?? [];

  return (
    <div className="flex gap-3 animate-fade-in">
      {/* Avatar */}
      <div className="w-8 h-8 rounded-full bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center shrink-0 mt-1">
        <Globe className="w-4 h-4 text-indigo-400" />
      </div>

      <div className="flex-1 min-w-0 space-y-3">
        {/* Style badges */}
        {styles.length > 0 && (
          <div className="flex items-center gap-2 flex-wrap">
            {styles.map((s) => (
              <span
                key={s}
                className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-500/15 border border-indigo-500/25 text-indigo-300"
              >
                {s}
              </span>
            ))}
          </div>
        )}

        {/* Destination cards grid */}
        {destinations.length > 0 && (
          <div
            className={clsx(
              "grid gap-3",
              destinations.length === 1 ? "grid-cols-1 max-w-xs" :
              destinations.length === 2 ? "grid-cols-2" :
              "grid-cols-2 sm:grid-cols-3"
            )}
          >
            {destinations.map((d) => (
              <DestinationCard key={d.name} meta={d} />
            ))}
          </div>
        )}

        {/* Message text / loading */}
        <div
          className={clsx(
            "px-4 py-3.5 rounded-2xl rounded-tl-sm text-sm leading-relaxed",
            "bg-[#161b2e] border border-[#1e2540]"
          )}
        >
          {message.loading && !message.text ? (
            <TypingIndicator />
          ) : (
            <div className="prose-travel">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.text}
              </ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
