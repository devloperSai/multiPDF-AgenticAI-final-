"""
Enhancement #3 — Summary Memory
--------------------------------
Problem (from your chat):
    "currently this system injecting long history to every prompt which is
     not an optimized way so we have to store summary of previous history
     and pass it with each LLM call"

Solution (from your chat):
    "Messages 1-N-3 → compressed into one summary paragraph (via LLM call)
     Last 3 messages → kept as raw conversation
     Both injected together into prompt
     This keeps context window small and relevant."

How it works:
    - Session has <= 6 messages total  → inject all as raw (no summary needed yet)
    - Session has >  6 messages total  → summarize everything except last 3,
                                         keep last 3 raw, inject both together

Why 6 as the cutoff:
    Below 6 messages the conversation is too short to summarize meaningfully.
    Summarizing a 2-turn exchange would produce a summary longer than the original.
    6 messages = 3 full Q&A turns — a reasonable minimum before compression kicks in.

Why last 3 raw:
    The most recent exchanges are what the LLM needs for immediate context
    (e.g. "what about that clause?" needs to see the previous answer).
    Summarizing them would lose the exact wording that coreference depends on.

Summary is generated once and cached in-memory per session.
Cache is invalidated when new messages arrive — so it rebuilds on next call.
This means: summary LLM call happens at most once per new message, not per query.
"""

from sqlalchemy.orm import Session as DBSession
from memory.chat_history import get_recent_messages, get_all_messages

# ── In-memory summary cache ───────────────────────────────────────────────────
# Structure: { session_id: { "message_count": int, "summary": str } }
# Invalidated when message count changes — rebuilt lazily on next request.
_summary_cache: dict = {}

# Number of recent messages to always keep as raw conversation
from config import cfg
RAW_TAIL = cfg.MEMORY_RAW_TAIL

# Only start summarizing when session has more than this many messages
SUMMARY_THRESHOLD = cfg.MEMORY_SUMMARY_THRESHOLD


def _get_summary_llm(text_to_summarize: str) -> str:
    """
    Call Groq (with Together AI fallback) to compress old conversation turns
    into a single concise summary paragraph.

    Uses low temperature (0.1) — we want factual compression, not creativity.
    Uses small max_tokens (300) — summary should be short by design.
    """
    from graph.fallback_llm import call_with_fallback

    messages = [
        {
            "role": "system",
            "content": (
                "You are a conversation summarizer. "
                "Compress the provided conversation history into a single concise paragraph. "
                "Preserve: key questions asked, key answers given, important facts mentioned, "
                "any document-specific details (names, dates, clauses, numbers). "
                "Drop: filler words, repeated information, pleasantries. "
                "Output only the summary paragraph — no preamble, no labels."
            )
        },
        {
            "role": "user",
            "content": f"Summarize this conversation history:\n\n{text_to_summarize}"
        }
    ]

    try:
        summary, provider = call_with_fallback(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=300
        )
        print(f"[memory] Summary generated via {provider} ({len(summary)} chars)")
        return summary.strip()
    except Exception as e:
        print(f"[memory] Summary LLM call failed: {e} — falling back to raw truncation")
        return text_to_summarize  # safe fallback: return original text if LLM fails


def _messages_to_text(messages: list) -> str:
    """Convert list of Message objects to readable text block."""
    lines = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def build_memory_context(db: DBSession, session_id: str, limit: int = 8) -> str:
    """
    Build conversation memory context to inject into LLM prompt.

    Short sessions (<= SUMMARY_THRESHOLD messages):
        Return last `limit` messages as raw text — no summarization needed.

    Long sessions (> SUMMARY_THRESHOLD messages):
        1. Fetch ALL messages for the session
        2. Split: old_messages = everything except last RAW_TAIL
                  recent_messages = last RAW_TAIL
        3. Summarize old_messages via LLM (cached — only regenerated when new messages arrive)
        4. Return: summary paragraph + raw recent messages

    Result injected into prompt as:
        [Summary of earlier conversation]
        <summary text>

        [Recent conversation]
        User: ...
        Assistant: ...
        User: ...

    This keeps the prompt context window lean regardless of how long the session grows.
    """
    # Get ALL messages to decide strategy
    all_msgs = get_all_messages(db, session_id)

    if not all_msgs:
        return ""

    # Short session — just return recent raw messages, no summary needed
    if len(all_msgs) <= SUMMARY_THRESHOLD:
        recent = all_msgs[-limit:]
        return _messages_to_text(recent)

    # Long session — summarize old, keep recent raw
    old_messages    = all_msgs[:-RAW_TAIL]   # everything except last 3
    recent_messages = all_msgs[-RAW_TAIL:]   # last 3 always raw

    # ── Summary cache check ───────────────────────────────────────────────────
    # Cache key: session_id + total message count
    # If message count changed (new message arrived), invalidate and regenerate
    cached = _summary_cache.get(session_id)
    if cached and cached["message_count"] == len(all_msgs):
        summary = cached["summary"]
        print(f"[memory] Using cached summary for session {session_id}")
    else:
        print(f"[memory] Generating new summary — {len(old_messages)} old messages to compress")
        old_text = _messages_to_text(old_messages)
        summary  = _get_summary_llm(old_text)

        # Store in cache with current message count as invalidation key
        _summary_cache[session_id] = {
            "message_count": len(all_msgs),
            "summary":       summary
        }
    # ─────────────────────────────────────────────────────────────────────────

    recent_text = _messages_to_text(recent_messages)

    return (
        f"[Summary of earlier conversation]\n{summary}"
        f"\n\n[Recent conversation]\n{recent_text}"
    )


def clear_summary_cache(session_id: str):
    """
    Clear cached summary for a session.
    Call this when a session is deleted so we don't leak memory.
    """
    if session_id in _summary_cache:
        del _summary_cache[session_id]
        print(f"[memory] Summary cache cleared for session {session_id}")