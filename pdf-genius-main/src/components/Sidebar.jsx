import { useApp } from "../context/AppContext";
import { Plus, MessageSquare, X, Trash2, Loader2 } from "lucide-react";
import { deleteSession } from "../lib/api";

export default function Sidebar({ open, onClose }) {
  const {
    sessions,
    setSessions,
    activeSessionId,
    setActiveSessionId,
    createSession,
    setMessages,
    loadingHistory,
  } = useApp();

  const handleNewChat = async () => {
    await createSession();
    onClose();
  };

  const handleSelectSession = (sessionId) => {
    setActiveSessionId(sessionId);
    onClose();
  };

  const handleDeleteSession = async (e, sessionId) => {
    e.stopPropagation();
    const ok = await deleteSession(sessionId);
    if (ok) {
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      setMessages((prev) => {
        const next = { ...prev };
        delete next[sessionId];
        return next;
      });
      if (activeSessionId === sessionId) setActiveSessionId(null);
    }
  };

  return (
    <>
      {open && (
        <div className="fixed inset-0 bg-foreground/20 z-40 lg:hidden" onClick={onClose} />
      )}
      <aside className={`
        fixed lg:static inset-y-0 left-0 z-50 w-64 bg-sidebar-bg flex flex-col
        transition-transform duration-200 lg:translate-x-0
        ${open ? "translate-x-0" : "-translate-x-full"}
      `}>
        <div className="flex items-center justify-between p-4">
          <span className="font-semibold text-sm" style={{ color: "hsl(var(--sidebar-fg))" }}>
            Chats
          </span>
          <button
            onClick={onClose}
            className="lg:hidden p-1 rounded hover:bg-sidebar-hover"
            style={{ color: "hsl(var(--sidebar-fg))" }}
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="px-3 mb-3">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border border-dashed text-sm font-medium transition-colors hover:bg-sidebar-hover"
            style={{ borderColor: "hsl(var(--sidebar-fg) / 0.2)", color: "hsl(var(--sidebar-fg))" }}
          >
            <Plus className="w-4 h-4" /> New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin px-3 space-y-1">
          {sessions.length === 0 && (
            <p className="text-xs text-center py-4" style={{ color: "hsl(var(--sidebar-fg) / 0.4)" }}>
              No sessions yet
            </p>
          )}

          {sessions.map((s) => (
            <div key={s.id} className="group relative">
              <button
                onClick={() => handleSelectSession(s.id)}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-left transition-colors pr-8 ${
                  s.id === activeSessionId ? "bg-sidebar-hover" : "hover:bg-sidebar-hover"
                }`}
                style={{
                  color:
                    s.id === activeSessionId
                      ? "hsl(var(--sidebar-active))"
                      : "hsl(var(--sidebar-fg))",
                }}
              >
                {/* Show spinner next to active session while its history loads */}
                {s.id === activeSessionId && loadingHistory ? (
                  <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin" />
                ) : (
                  <MessageSquare className="w-3.5 h-3.5 shrink-0" />
                )}
                <span className="truncate">{s.title}</span>
              </button>

              {/* Delete button — appears on hover */}
              <button
                onClick={(e) => handleDeleteSession(e, s.id)}
                className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-destructive/20"
              >
                <Trash2 className="w-3 h-3 text-muted-foreground hover:text-destructive" />
              </button>
            </div>
          ))}
        </div>
      </aside>
    </>
  );
}