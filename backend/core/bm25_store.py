"""
core/bm25_store.py

Manages in-memory BM25 indexes per session.
BM25 index is built on first query and cached for the session lifetime.
Rebuilt automatically when new chunks are added (store_chunks called).
"""

import re
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi

# In-memory cache: session_id -> {"index": BM25Okapi, "chunks": [...]}
_bm25_cache: Dict[str, Dict] = {}


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    Lowercases, removes punctuation, splits on whitespace.
    Keeps numbers intact — important for legal/financial docs.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]  # remove single chars


def build_bm25_index(session_id: str, chunks: List[Dict]) -> None:
    """
    Build and cache a BM25 index for a session from a list of chunks.
    Called after store_chunks in pdf_worker.py.

    chunks: list of dicts with keys: text, metadata, chunk_id (same format as vector_store)
    """
    if not chunks:
        return

    tokenized = [_tokenize(c["text"]) for c in chunks]
    index = BM25Okapi(tokenized)

    _bm25_cache[session_id] = {
        "index":  index,
        "chunks": chunks,   # keep original chunks for metadata lookup
    }
    print(f"[bm25] Built index for session {session_id} — {len(chunks)} chunks")


def get_bm25_index(session_id: str) -> Optional[Dict]:
    """Return cached BM25 index entry or None if not built yet."""
    return _bm25_cache.get(session_id)


def invalidate_bm25_index(session_id: str) -> None:
    """
    Invalidate cached index for a session.
    Called when new PDF is uploaded or session is deleted
    so index gets rebuilt fresh on next query.
    """
    if session_id in _bm25_cache:
        del _bm25_cache[session_id]
        print(f"[bm25] Cache invalidated for session {session_id}")


def bm25_search(
    session_id: str,
    query: str,
    top_k: int = 10,
    pdf_id: Optional[str] = None
) -> List[Dict]:
    """
    BM25 keyword search for a session.
    Returns top_k chunks ranked by BM25 score.
    Optionally filter by pdf_id for comparison intent.

    Returns same format as query_chunks:
    [{"text": ..., "metadata": ..., "score": ...}]
    """
    cached = get_bm25_index(session_id)

    if not cached:
        # Index not built yet — load chunks from ChromaDB and build
        cached = _load_and_build(session_id)
        if not cached:
            print(f"[bm25] No chunks found for session {session_id}")
            return []

    index  = cached["index"]
    chunks = cached["chunks"]

    # Filter by pdf_id if requested (comparison intent)
    if pdf_id:
        filtered_chunks = [c for c in chunks if c.get("metadata", {}).get("pdf_id") == pdf_id]
        if not filtered_chunks:
            return []
        tokenized = [_tokenize(c["text"]) for c in filtered_chunks]
        local_index = BM25Okapi(tokenized)
        scores = local_index.get_scores(_tokenize(query))
        chunks_to_rank = filtered_chunks
    else:
        scores = index.get_scores(_tokenize(query))
        chunks_to_rank = chunks

    # Pair chunks with scores, sort descending
    scored = sorted(
        zip(scores, chunks_to_rank),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    results = []
    for score, chunk in scored:
        if score <= 0:
            continue  # skip zero-score chunks — no keyword overlap at all
        results.append({
            "text":     chunk["text"],
            "metadata": chunk["metadata"],
            "score":    round(float(score), 4),  # raw BM25 score
            "bm25_score": round(float(score), 4),
        })

    print(f"[bm25] Query '{query[:40]}...' → {len(results)} results")
    return results


def _load_and_build(session_id: str) -> Optional[Dict]:
    """
    Load all chunks from ChromaDB for a session and build BM25 index.
    Called lazily on first BM25 query if index not pre-built.
    """
    try:
        from core.vector_store import _get_collection

        collection = _get_collection(session_id)
        count = collection.count()

        if count == 0:
            return None

        # Fetch all chunks from ChromaDB
        results = collection.get(include=["documents", "metadatas"])
        chunks = []
        for i, doc in enumerate(results["documents"]):
            chunks.append({
                "text":     doc,
                "metadata": results["metadatas"][i],
            })

        build_bm25_index(session_id, chunks)
        return _bm25_cache.get(session_id)

    except Exception as e:
        print(f"[bm25] Failed to load and build index: {e}")
        return None