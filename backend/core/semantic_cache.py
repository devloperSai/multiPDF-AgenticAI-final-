import numpy as np
from typing import Optional
from collections import OrderedDict

# In-memory cache — stores per session
# Structure: { session_id: OrderedDict{ question: {embedding, answer, citations} } }
# Refine 5 — OrderedDict gives true LRU: move_to_end on access, popitem(last=False) on evict
_cache: dict = {}

SIMILARITY_THRESHOLD = 0.92
MAX_CACHE_SIZE = 100  # max entries per session — raised from 50 to 100


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_cached_answer(
    session_id: str,
    question: str,
    question_embedding: list
) -> Optional[dict]:
    """
    Check if a similar question has been answered before.
    Returns cached result if similarity > threshold, else None.
    Refine 5 — on cache hit, move entry to end (most recently used).
    """
    session_cache = _cache.get(session_id)
    if not session_cache:
        return None

    best_score = 0.0
    best_key = None
    best_entry = None

    for key, entry in session_cache.items():
        score = _cosine_similarity(question_embedding, entry["embedding"])
        if score > best_score:
            best_score = score
            best_key = key
            best_entry = entry

    if best_score >= SIMILARITY_THRESHOLD:
        # Refine 5 — promote to most recently used
        session_cache.move_to_end(best_key)
        print(f"[cache] HIT — similarity={best_score:.4f} | original='{best_entry['question']}'")
        return {
            "answer": best_entry["answer"],
            "citations": best_entry["citations"],
            "cache_hit": True,
            "cache_similarity": round(best_score, 4),
            "original_question": best_entry["question"]
        }

    print(f"[cache] MISS — best similarity={best_score:.4f}")
    return None


def store_in_cache(
    session_id: str,
    question: str,
    question_embedding: list,
    answer: str,
    citations: list
):
    """
    Store a question-answer pair in the semantic cache.
    Refine 5 — evicts least recently used entry when full.
    """
    if not answer or "failed" in answer.lower() or "not available" in answer.lower():
        return

    if not question_embedding:
        return

    if session_id not in _cache:
        _cache[session_id] = OrderedDict()

    session_cache = _cache[session_id]

    # If question already cached — update it and promote to most recent
    if question in session_cache:
        session_cache[question]["answer"] = answer
        session_cache[question]["citations"] = citations
        session_cache.move_to_end(question)
        print(f"[cache] Updated existing entry for session {session_id}")
        return

    # Refine 5 — evict least recently used (front of OrderedDict) when at capacity
    if len(session_cache) >= MAX_CACHE_SIZE:
        evicted_key, _ = session_cache.popitem(last=False)
        print(f"[cache] LRU evicted: '{evicted_key[:50]}...' for session {session_id}")

    session_cache[question] = {
        "question": question,
        "embedding": question_embedding,
        "answer": answer,
        "citations": citations
    }

    print(f"[cache] Stored — session {session_id} | cache size: {len(session_cache)}")


def clear_cache(session_id: str):
    """Clear cache for a session — called when new PDF is uploaded."""
    if session_id in _cache:
        del _cache[session_id]
        print(f"[cache] Cleared for session {session_id}")


def get_cache_stats(session_id: str) -> dict:
    """Return cache stats for a session."""
    session_cache = _cache.get(session_id, OrderedDict())
    entries = list(session_cache.keys())
    return {
        "session_id": session_id,
        "cached_questions": len(session_cache),
        "max_size": MAX_CACHE_SIZE,
        "questions": entries
    }