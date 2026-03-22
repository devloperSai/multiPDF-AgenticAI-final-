import { createContext, useContext, useState, useCallback, useEffect } from "react";
import { createSessionOnBackend, fetchSessions, fetchSessionMessages } from "../lib/api";

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const stored = localStorage.getItem("user");
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState({});
  const [isStreaming, setIsStreaming] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const activeMessages = messages[activeSessionId] || [];

  // ── Load sessions from backend on login ───────────────────────────────────
  useEffect(() => {
    if (!user) return;
    fetchSessions().then((data) => {
      if (Array.isArray(data) && data.length > 0) {
        const mapped = data.map((s) => ({
          id: s.session_id,
          title: s.title || "New Chat",
          updatedAt: s.updated_at,
        }));
        setSessions(mapped);
      }
    });
  }, [user]);

  // ── Load chat history when active session changes ─────────────────────────
  useEffect(() => {
    if (!activeSessionId) return;
    if (messages[activeSessionId] !== undefined) return; // already loaded

    setLoadingHistory(true);
    fetchSessionMessages(activeSessionId)
      .then((data) => {
        if (!Array.isArray(data) || data.length === 0) {
          setMessages((prev) => ({ ...prev, [activeSessionId]: [] }));
          return;
        }
        const mapped = data.map((m) => ({
          role: m.role === "assistant" ? "ai" : "user",
          text: m.content,
          citations: m.citations || [],
          // Historical messages don't carry live metadata
          intent: null,
          cacheHit: false,
          hydeUsed: false,
          retrievalConfidence: null,
        }));
        setMessages((prev) => ({ ...prev, [activeSessionId]: mapped }));
      })
      .catch(() => {
        setMessages((prev) => ({ ...prev, [activeSessionId]: [] }));
      })
      .finally(() => setLoadingHistory(false));
  }, [activeSessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── messages ──────────────────────────────────────────────────────────────

  const addMessage = useCallback((sessionId, message) => {
    setMessages((prev) => ({
      ...prev,
      [sessionId]: [...(prev[sessionId] || []), message],
    }));
  }, []);

  const updateLastMessage = useCallback((sessionId, text) => {
    setMessages((prev) => {
      const msgs = [...(prev[sessionId] || [])];
      if (msgs.length > 0) {
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], text };
      }
      return { ...prev, [sessionId]: msgs };
    });
  }, []);

  const updateLastMessageCitations = useCallback((sessionId, citations) => {
    setMessages((prev) => {
      const msgs = [...(prev[sessionId] || [])];
      if (msgs.length > 0) {
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], citations };
      }
      return { ...prev, [sessionId]: msgs };
    });
  }, []);

  // Attach intent / cache_hit / hyde_used / retrieval_confidence to last AI message
  // Called once when the "done" SSE event fires
  const updateLastMessageMeta = useCallback((sessionId, meta) => {
    setMessages((prev) => {
      const msgs = [...(prev[sessionId] || [])];
      if (msgs.length > 0) {
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], ...meta };
      }
      return { ...prev, [sessionId]: msgs };
    });
  }, []);

  // ── sessions ──────────────────────────────────────────────────────────────

  const createSession = useCallback(async () => {
    const backendId = await createSessionOnBackend();
    const id = backendId || "local_" + Date.now();
    const session = { id, title: "New Chat", updatedAt: new Date().toISOString() };
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(id);
    setMessages((prev) => ({ ...prev, [id]: [] }));
    return id;
  }, []);

  const updateSessionTitle = useCallback((sessionId, title) => {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === sessionId
          ? { ...s, title: title.slice(0, 40) + (title.length > 40 ? "…" : "") }
          : s
      )
    );
  }, []);

  // ── auth ──────────────────────────────────────────────────────────────────

  const login = useCallback((userData) => {
    setUser(userData);
    localStorage.setItem("user", JSON.stringify(userData));
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    setSessions([]);
    setMessages({});
    setActiveSessionId(null);
    localStorage.removeItem("user");
    localStorage.removeItem("token");
  }, []);

  return (
    <AppContext.Provider
      value={{
        user, setUser, login, logout,
        sessions, setSessions,
        activeSessionId, setActiveSessionId,
        activeMessages,
        addMessage,
        updateLastMessage,
        updateLastMessageCitations,
        updateLastMessageMeta,
        updateSessionTitle,
        createSession,
        isStreaming, setIsStreaming,
        loadingHistory,
        uploadStatus, setUploadStatus,
        messages, setMessages,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}