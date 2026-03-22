from sentence_transformers import CrossEncoder
from typing import List, Dict

_model: CrossEncoder = None

# ── Enhancement #7: Reranker Score Threshold ──────────────────────────────────
# Calibrated from real observed scores on this system:
#   Irrelevant question scores  →  around -11.1 to -11.2
#   Borderline relevant scores  →  around  -1.8 to  -3.8
#   Clearly relevant scores     →  around  +3.0 to +15.0
#
#   -8.0 sits safely between irrelevant (-11) and borderline (-3.8):
#     above -8   →  at least weakly relevant — KEEP
#     below -8   →  confidently irrelevant   — DISCARD
#
# IMPORTANT: threshold is NOT applied for summary/comparison intents.
# Those queries are vague by nature — cross-encoder scores all chunks
# low (~-6 to -8) because no chunk "answers" a summary request directly.
# Threshold on summary would discard ALL chunks → empty answer.
# skip_threshold=True bypasses this filter for those intents.
from config import cfg
RERANKER_SCORE_THRESHOLD = cfg.RERANKER_THRESHOLD


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print("[reranker] Loading cross-encoder/ms-marco-MiniLM-L-6-v2...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[reranker] Model loaded.")
    return _model


def rerank_chunks(
    question:       str,
    chunks:         List[Dict],
    top_k:          int = 6,
    skip_threshold: bool = False
) -> List[Dict]:
    """
    Re-score chunks using cross-encoder relevance scoring.

    Cross-encoder reads question + chunk together — understands actual relevance,
    much more accurately than cosine similarity alone.

    Args:
        question:       The user's question.
        chunks:         Retrieved chunks to rerank.
        top_k:          Max chunks to return.
        skip_threshold: If True, skip the score threshold filter.
                        Set True for summary/comparison intents where vague
                        queries naturally score all chunks low.

    Returns up to top_k chunks, sorted by cross-encoder score descending.
    """
    if not chunks:
        return []

    model = _get_model()

    # Build pairs: [question, chunk_text] for each chunk
    pairs = [[question, c["text"]] for c in chunks]

    # Score all pairs in one batch
    scores = model.predict(pairs)

    # Attach cross-encoder score to each chunk
    for i, chunk in enumerate(chunks):
        chunk["cross_score"] = float(scores[i])

    # Sort by score descending — best chunks first
    reranked = sorted(chunks, key=lambda x: x["cross_score"], reverse=True)

    # ── Threshold filter ──────────────────────────────────────────────────────
    if skip_threshold:
        print(f"[reranker] Threshold skipped — returning top {min(top_k, len(reranked))} of {len(reranked)} chunks")
    else:
        before = len(reranked)
        reranked = [c for c in reranked if c["cross_score"] >= RERANKER_SCORE_THRESHOLD]
        removed = before - len(reranked)
        if removed:
            print(f"[reranker] Threshold filter (-8): removed {removed} irrelevant chunks")
    # ─────────────────────────────────────────────────────────────────────────

    result = reranked[:top_k]

    print(f"[reranker] {len(chunks)} chunks → {len(reranked)} after filter → top {len(result)} returned")
    for i, c in enumerate(result[:3]):
        print(f"  [{i+1}] cross_score={c['cross_score']:.4f} | rrf_score={c.get('score', 0):.4f}")

    return result