import { useState } from "react";
import { Send, ChevronUp, Zap, AlignLeft, BookOpen, List } from "lucide-react";

// Response mode config — maps mode key to display info
const MODES = [
  {
    key:   null,
    label: "Auto",
    icon:  Zap,
    desc:  "Let AI decide format",
    color: "text-primary",
    bg:    "bg-primary/10 border-primary/20",
  },
  {
    key:   "short",
    label: "Short",
    icon:  ChevronUp,
    desc:  "Brief, 2-3 sentences",
    color: "text-amber-400",
    bg:    "bg-amber-500/10 border-amber-500/20",
  },
  {
    key:   "explanation",
    label: "Explain",
    icon:  BookOpen,
    desc:  "Detailed explanation",
    color: "text-blue-400",
    bg:    "bg-blue-500/10 border-blue-500/20",
  },
  {
    key:   "bullets",
    label: "Bullets",
    icon:  List,
    desc:  "Structured list format",
    color: "text-emerald-400",
    bg:    "bg-emerald-500/10 border-emerald-500/20",
  },
  {
    key:   "verbatim",
    label: "Verbatim",
    icon:  AlignLeft,
    desc:  "Exact text from document",
    color: "text-violet-400",
    bg:    "bg-violet-500/10 border-violet-500/20",
  },
];

export default function InputBar({ onSend, disabled }) {
  const [text,        setText]        = useState("");
  const [mode,        setMode]        = useState(null);   // null = Auto
  const [showModes,   setShowModes]   = useState(false);

  const activeMode = MODES.find((m) => m.key === mode) || MODES[0];
  const ActiveIcon = activeMode.icon;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() || disabled) return;
    onSend(text.trim(), mode);   // pass mode to ChatWindow
    setText("");
    setShowModes(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
    if (e.key === "Escape") setShowModes(false);
  };

  const selectMode = (key) => {
    setMode(key);
    setShowModes(false);
  };

  return (
    <div className="relative border-t border-border bg-card">
      {/* Mode picker popup — appears above input bar */}
      {showModes && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setShowModes(false)}
          />
          <div className="absolute bottom-full left-4 mb-2 z-20 bg-card border border-border rounded-xl shadow-elevated p-1.5 min-w-[200px] animate-fade-in-up">
            <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider px-2 py-1">
              Response Format
            </p>
            {MODES.map((m) => {
              const Icon = m.icon;
              const isActive = mode === m.key;
              return (
                <button
                  key={m.key ?? "auto"}
                  onClick={() => selectMode(m.key)}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left transition-colors ${
                    isActive ? m.bg + " border" : "hover:bg-muted"
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 shrink-0 ${isActive ? m.color : "text-muted-foreground"}`} />
                  <div>
                    <p className={`text-xs font-medium ${isActive ? m.color : "text-foreground"}`}>
                      {m.label}
                    </p>
                    <p className="text-[10px] text-muted-foreground">{m.desc}</p>
                  </div>
                  {isActive && (
                    <div className={`ml-auto w-1.5 h-1.5 rounded-full ${m.color.replace("text-", "bg-")}`} />
                  )}
                </button>
              );
            })}
          </div>
        </>
      )}

      {/* Input row */}
      <form onSubmit={handleSubmit} className="flex items-center gap-2 p-3 px-4">

        {/* Mode toggle button */}
        <button
          type="button"
          onClick={() => setShowModes((v) => !v)}
          disabled={disabled}
          title={`Format: ${activeMode.label}`}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-medium transition-colors shrink-0 disabled:opacity-40 ${
            mode
              ? activeMode.bg + " " + activeMode.color
              : "border-border text-muted-foreground hover:bg-muted"
          }`}
        >
          <ActiveIcon className="w-3 h-3" />
          <span className="hidden sm:inline">{activeMode.label}</span>
        </button>

        {/* Text input */}
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Ask a question about your PDFs..."
          className="flex-1 px-4 py-2.5 rounded-lg border border-input bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring text-sm transition-colors"
        />

        {/* Send button */}
        <button
          type="submit"
          disabled={disabled || !text.trim()}
          className="p-2.5 rounded-lg gradient-primary text-primary-foreground hover:opacity-90 transition-opacity disabled:opacity-40 shrink-0"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}