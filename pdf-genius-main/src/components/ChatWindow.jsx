import { useRef, useEffect, useCallback } from "react";
import { useApp } from "../context/AppContext";
import MessageBubble from "./MessageBubble";
import InputBar from "./InputBar";
import { streamAnswer } from "../lib/api";
import { MessageSquare, Loader2 } from "lucide-react";

// ── Streaming buffer ───────────────────────────────────────────────────────
// Problem: Groq streams tokens one char at a time (or small chunks).
// Each token triggers a React state update → re-render → visible jitter.
// Claude-like smooth streaming: buffer tokens and flush every 30ms.
//
// This gives:
//   - Smooth word-by-word appearance (not char-by-char flicker)
//   - Fewer React re-renders (batched updates instead of one per char)
//   - Natural reading pace without losing streaming feel
const FLUSH_INTERVAL_MS = 30;  // flush buffer every 30ms — smooth but responsive

export default function ChatWindow() {
  const {
    activeSessionId,
    activeMessages,
    addMessage,
    updateLastMessage,
    updateLastMessageCitations,
    updateLastMessageMeta,
    updateSessionTitle,
    isStreaming,
    setIsStreaming,
    createSession,
    loadingHistory,
  } = useApp();

  const bottomRef      = useRef();
  const bufferRef      = useRef("");      // pending tokens not yet flushed to state
  const flushTimerRef  = useRef(null);    // setInterval handle
  const accumulatedRef = useRef("");      // full accumulated text (for error fallback)
  const sessionIdRef   = useRef(null);    // current streaming session

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeMessages]);

  // ── Buffer flush logic ─────────────────────────────────────────────────
  const startFlushTimer = useCallback((sessionId) => {
    if (flushTimerRef.current) return; // already running
    flushTimerRef.current = setInterval(() => {
      if (bufferRef.current) {
        updateLastMessage(sessionId, accumulatedRef.current);
        bufferRef.current = "";
      }
    }, FLUSH_INTERVAL_MS);
  }, [updateLastMessage]);

  const stopFlushTimer = useCallback((sessionId) => {
    if (flushTimerRef.current) {
      clearInterval(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    // Final flush — ensure all buffered text is shown
    if (bufferRef.current) {
      updateLastMessage(sessionId, accumulatedRef.current);
      bufferRef.current = "";
    }
  }, [updateLastMessage]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (flushTimerRef.current) clearInterval(flushTimerRef.current);
    };
  }, []);

  // ── Handle send ────────────────────────────────────────────────────────
  const handleSend = async (text, responseMode = null) => {
    let sessionId = activeSessionId;
    if (!sessionId) sessionId = await createSession();
    if (!sessionId) return;

    // Reset buffers for new message
    bufferRef.current     = "";
    accumulatedRef.current = "";
    sessionIdRef.current  = sessionId;

    updateSessionTitle(sessionId, text);

    addMessage(sessionId, { role: "user", text });
    addMessage(sessionId, {
      role:               "ai",
      text:               "",
      citations:          [],
      intent:             null,
      cacheHit:           false,
      hydeUsed:           false,
      retrievalConfidence: null,
    });
    setIsStreaming(true);

    // Start buffer flush timer
    startFlushTimer(sessionId);

    streamAnswer(
      text,
      sessionId,
      // onToken — add to buffer, don't update state directly
      (token) => {
        accumulatedRef.current += token;
        bufferRef.current      += token;
        // Don't call updateLastMessage here — let the timer batch it
      },
      // onDone
      () => {
        stopFlushTimer(sessionId);
        setIsStreaming(false);
      },
      // onError
      (errMsg) => {
        stopFlushTimer(sessionId);
        updateLastMessage(sessionId, accumulatedRef.current || `⚠️ Error: ${errMsg}`);
        setIsStreaming(false);
      },
      // onCitations
      (citations) => updateLastMessageCitations(sessionId, citations),
      // onMeta
      (meta) => {
        updateLastMessageMeta(sessionId, {
          intent:              meta.query_intent     || null,
          cacheHit:            meta.cache_hit        || false,
          hydeUsed:            meta.hyde_used        || false,
          retrievalConfidence: meta.retrieval_confidence ?? null,
        });
      },
      // responseMode — passed to API
      responseMode,
    );
  };

  if (loadingHistory) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 text-muted-foreground">
        <Loader2 className="w-6 h-6 animate-spin text-primary" />
        <p className="text-sm">Loading chat history...</p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
        {activeMessages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center h-full text-center py-20">
            <div className="w-14 h-14 rounded-2xl gradient-primary flex items-center justify-center mb-4">
              <MessageSquare className="w-7 h-7 text-primary-foreground" />
            </div>
            <h2 className="text-lg font-semibold text-foreground mb-1">Start a conversation</h2>
            <p className="text-sm text-muted-foreground max-w-sm">
              Upload a PDF and ask questions. AI will search through your documents and provide answers with citations.
            </p>
          </div>
        ) : (
          activeMessages.map((msg, i) => <MessageBubble key={i} message={msg} />)
        )}
        {isStreaming && (
          <div className="flex items-center gap-1.5 pl-10">
            <div className="w-1.5 h-1.5 rounded-full bg-primary typing-dot" />
            <div className="w-1.5 h-1.5 rounded-full bg-primary typing-dot" />
            <div className="w-1.5 h-1.5 rounded-full bg-primary typing-dot" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <InputBar onSend={handleSend} disabled={isStreaming} />
    </div>
  );
}