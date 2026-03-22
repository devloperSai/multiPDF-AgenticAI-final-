"""
Coreference Resolution — Question Rewriter
-------------------------------------------
Rewrites ambiguous follow-up questions to be self-contained
before they are embedded for retrieval.

Problem:
    User: "What are the termination clauses?"
    AI:   "The Institute can terminate with 30 days notice..."
    User: "What happens if they violate it?"
          ↑ "they" = Service Provider, "it" = termination clause
          But the embedder sees "they" and "it" — retrieves wrong chunks.

Solution:
    Detect reference words → rewrite using conversation history → embed rewritten question.
    Original question preserved for display, DB storage, and cache.
    Rewritten question used only for retrieval (embedding + BM25).

Fallback chain (per our rule — never crash, never degrade):
    1. LLM rewrite via call_with_fallback (Groq → Together → ...)
       → if all providers fail → use original question unchanged
    2. If no conversation history → skip rewrite (nothing to resolve against)
    3. If question has no reference words → skip rewrite (save API call)

Cost:
    ~80 input tokens + ~20 output tokens per triggered rewrite.
    Only triggered when BOTH conditions are true:
      a) question contains reference words (they, it, this, that, etc.)
      b) session has at least 2 prior messages to resolve against
    Estimated trigger rate: ~20-30% of questions in active sessions.
"""

import re
from typing import Optional
from config import cfg

# ── Reference word detection ──────────────────────────────────────────────────
# Words that indicate the question refers to something from prior context.
# Short words only checked at word boundaries to avoid false positives.
# e.g. "it" should not match "institute" or "item".
REFERENCE_WORDS = {
    # Pronouns
    "it", "its", "they", "them", "their", "theirs",
    "he", "she", "his", "her", "hers",
    # Demonstratives
    "this", "that", "these", "those",
    # Reference phrases (checked as substrings — longer, safe)
    "the same", "the above", "such", "aforementioned",
    "mentioned", "said clause", "said contract",
    "the clause", "the section", "the article",
}

# Short words (<=4 chars) — check at word boundary to avoid false positives
SHORT_REFS = {"it", "its", "he", "she", "his", "her", "they", "them",
              "this", "that"}

# Longer phrases — simple substring check is safe
PHRASE_REFS = REFERENCE_WORDS - SHORT_REFS


def _has_reference(question: str) -> bool:
    """
    Check if question contains reference words that need resolution.
    Returns True if rewrite is needed, False if question is self-contained.
    """
    q = question.lower().strip()

    # Check short words at word boundaries
    for word in SHORT_REFS:
        if re.search(rf'\b{re.escape(word)}\b', q):
            return True

    # Check longer phrases as substrings
    for phrase in PHRASE_REFS:
        if phrase in q:
            return True

    return False


def _build_history_text(messages: list, limit: int = 4) -> str:
    """
    Build compact conversation history for the rewriter prompt.
    Takes last `limit` messages (2 Q&A pairs = 4 messages).
    Truncates long AI answers to 200 chars — rewriter only needs gist.
    """
    recent = messages[-limit:] if len(messages) > limit else messages
    lines  = []
    for msg in recent:
        role    = "User" if msg.role == "user" else "Assistant"
        content = msg.content
        if msg.role == "assistant" and len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def resolve_coreferences(
    question:   str,
    session_id: str,
    db
) -> str:
    """
    Rewrite question to be self-contained using conversation history.

    Args:
        question:   The user's current question (may contain pronouns)
        session_id: Session ID to fetch conversation history from
        db:         SQLAlchemy DB session

    Returns:
        Rewritten question if references detected + history available.
        Original question unchanged in all fallback cases.
    """
    # ── Step 1: Quick check — does question need rewriting? ───────────────────
    if not _has_reference(question):
        print(f"[coref] No references detected — skipping rewrite")
        return question

    # ── Step 2: Get conversation history ─────────────────────────────────────
    try:
        from memory.chat_history import get_recent_messages
        messages = get_recent_messages(db, session_id, limit=6)
    except Exception as e:
        print(f"[coref] Failed to fetch history: {e} — using original question")
        return question

    # Need at least 2 prior messages (1 Q&A pair) to resolve against
    if len(messages) < 2:
        print(f"[coref] Not enough history ({len(messages)} messages) — skipping rewrite")
        return question

    history_text = _build_history_text(messages)

    # ── Step 3: LLM rewrite ───────────────────────────────────────────────────
    try:
        from graph.fallback_llm import call_with_fallback

        rewritten, provider = call_with_fallback(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a question rewriter for a document QA system. "
                        "Your job: rewrite the current question to be fully self-contained "
                        "by replacing all pronouns and references with their actual subjects "
                        "from the conversation history.\n\n"
                        "Rules:\n"
                        "- Return ONLY the rewritten question — no explanation, no preamble\n"
                        "- Keep the question concise — do not add unnecessary words\n"
                        "- If the question is already self-contained, return it unchanged\n"
                        "- Never answer the question — only rewrite it"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_text}\n\n"
                        f"Current question: {question}\n\n"
                        f"Rewritten question:"
                    )
                }
            ],
            model=cfg.PRIMARY_LLM_MODEL,
            temperature=0.0,   # deterministic — rewriting is not creative
            max_tokens=80,     # rewritten question should be short
        )

        rewritten = rewritten.strip().strip('"').strip("'")

        # Sanity check — if rewritten is way longer than original, something went wrong
        if len(rewritten) > len(question) * 3:
            print(f"[coref] Rewritten question suspiciously long — using original")
            return question

        # If LLM returned basically the same question, no rewrite needed
        if rewritten.lower() == question.lower():
            print(f"[coref] No change from rewrite — question was already self-contained")
            return question

        print(f"[coref] Rewritten via {provider}:")
        print(f"  Original:  {question}")
        print(f"  Rewritten: {rewritten}")
        return rewritten

    except Exception as e:
        # All providers failed — use original question, retrieval continues
        print(f"[coref] All providers failed: {e} — using original question")
        return question