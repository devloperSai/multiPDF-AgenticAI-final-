from sentence_transformers import CrossEncoder
from typing import List, Dict

_model: CrossEncoder = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print("[reranker] Loading cross-encoder/ms-marco-MiniLM-L-6-v2...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[reranker] Model loaded.")
    return _model


def rerank_chunks(question: str, chunks: List[Dict], top_k: int = 6) -> List[Dict]:
    """
    Re-score chunks using cross-encoder relevance scoring.
    
    Cross-encoder reads question + chunk together — understands actual relevance.
    Much more accurate than cosine similarity alone.
    
    Returns top_k chunks sorted by cross-encoder score descending.
    """
    if not chunks:
        return []

    model = _get_model()

    # Build pairs: [question, chunk_text] for each chunk
    pairs = [[question, c["text"]] for c in chunks]

    # Score all pairs
    scores = model.predict(pairs)

    # Attach cross-encoder score to each chunk
    for i, chunk in enumerate(chunks):
        chunk["cross_score"] = float(scores[i])

    # Sort by cross-encoder score descending
    reranked = sorted(chunks, key=lambda x: x["cross_score"], reverse=True)

    # Return top_k
    result = reranked[:top_k]

    print(f"[reranker] {len(chunks)} chunks → top {len(result)} after reranking")
    for i, c in enumerate(result[:3]):
        print(f"  [{i+1}] cross_score={c['cross_score']:.4f} | cosine={c['score']:.4f}")

    return result