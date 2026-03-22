import numpy as np
from typing import Optional
from collections import OrderedDict

# In-memory cache — stores per session
# Structure: { session_id: OrderedDict{ "question||intent": {embedding, answer, citations, intent} } }
# Intent is part of the key — same question with different intent never shares a cache entry
_cache: dict = {}

SIMILARITY_THRESHOLD = 0.92
MAX_CACHE_SIZE = 100


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _make_cache_key(question: str, intent: str, response_mode: str = None) -> str:
    """
    Composite cache key = question + intent + response_mode.

    Three dimensions:
      - intent:        same question as factual vs summary = different retrieval
      - response_mode: same question with bullets vs explanation = different format

    Examples:
      "what are termination clauses" + factual + None    → one entry (Auto)
      "what are termination clauses" + factual + bullets → separate entry
      "what are termination clauses" + factual + explanation → separate entry

    This ensures changing the response mode always generates a fresh answer
    instead of returning the cached formatted answer from a previous mode.
    """
    mode_part = response_mode or "auto"
    return f"{question}||{intent or 'factual'}||{mode_part}"


def get_cached_answer(
    session_id: str,
    question: str,
    question_embedding: list,
    intent: str = "factual",
    response_mode: str = None
) -> Optional[dict]:
    """
    Check if a similar question with same intent AND response_mode was answered.
    Returns cached result if:
      1. Intent matches exactly
      2. response_mode matches exactly
      3. Embedding similarity > SIMILARITY_THRESHOLD
    All three must match — mode mismatch = always cache miss.
    """
    session_cache = _cache.get(session_id)
    if not session_cache:
        return None

    best_score = 0.0
    best_key   = None
    best_entry = None

    for key, entry in session_cache.items():
        # ── Intent must match exactly ──
        if entry.get("intent", "factual") != (intent or "factual"):
            continue
        # ── response_mode must match exactly ──
        if entry.get("response_mode") != response_mode:
            continue

        score = _cosine_similarity(question_embedding, entry["embedding"])
        if score > best_score:
            best_score = score
            best_key   = key
            best_entry = entry

    if best_score >= SIMILARITY_THRESHOLD:
        session_cache.move_to_end(best_key)
        print(f"[cache] HIT — intent={intent} | mode={response_mode} | similarity={best_score:.4f}")
        return {
            "answer":            best_entry["answer"],
            "citations":         best_entry["citations"],
            "cache_hit":         True,
            "cache_similarity":  round(best_score, 4),
            "original_question": best_entry["question"]
        }

    print(f"[cache] MISS — intent={intent} | mode={response_mode} | best similarity={best_score:.4f}")
    return None


def store_in_cache(
    session_id: str,
    question: str,
    question_embedding: list,
    answer: str,
    citations: list,
    intent: str = "factual",
    response_mode: str = None
):
    """
    Store a question-answer pair with intent + response_mode in the cache.
    Same question stored separately per (intent, response_mode) combination.
    Changing mode always generates a fresh answer.
    """
    if not answer or "failed" in answer.lower() or "not available" in answer.lower():
        return

    if not question_embedding:
        return

    if session_id not in _cache:
        _cache[session_id] = OrderedDict()

    session_cache = _cache[session_id]
    cache_key     = _make_cache_key(question, intent, response_mode)

    # Update existing entry if same question + same intent + same mode
    if cache_key in session_cache:
        session_cache[cache_key]["answer"]    = answer
        session_cache[cache_key]["citations"] = citations
        session_cache.move_to_end(cache_key)
        print(f"[cache] Updated — intent={intent} | mode={response_mode} | session {session_id}")
        return

    # Evict LRU entry when at capacity
    if len(session_cache) >= MAX_CACHE_SIZE:
        evicted_key, _ = session_cache.popitem(last=False)
        print(f"[cache] LRU evicted: '{evicted_key[:50]}' for session {session_id}")

    session_cache[cache_key] = {
        "question":     question,
        "intent":       intent or "factual",
        "response_mode": response_mode,
        "embedding":    question_embedding,
        "answer":       answer,
        "citations":    citations
    }

    print(f"[cache] Stored — intent={intent} | mode={response_mode} | session {session_id} | cache size: {len(session_cache)}")


def clear_cache(session_id: str):
    """Clear cache for a session — called when new PDF is uploaded."""
    if session_id in _cache:
        del _cache[session_id]
        print(f"[cache] Cleared for session {session_id}")


def get_cache_stats(session_id: str) -> dict:
    """Return cache stats for a session."""
    session_cache = _cache.get(session_id, OrderedDict())
    return {
        "session_id":        session_id,
        "cached_questions":  len(session_cache),
        "max_size":          MAX_CACHE_SIZE,
        "entries":           [
            {"key": k, "intent": v.get("intent"), "question": v.get("question")}
            for k, v in session_cache.items()
        ]
    }