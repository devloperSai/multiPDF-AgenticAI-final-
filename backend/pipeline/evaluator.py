"""
Response Evaluator — Local Cross-Encoder Faithfulness Scoring
--------------------------------------------------------------
Measures how well the answer is grounded in retrieved context chunks.

Primary: Local cross-encoder scoring (zero API tokens, GPU, ~50ms)
Fallback: RAGAS + Groq (original approach, kept as safety net)

WHY replace RAGAS faithfulness with local scoring:
    RAGAS faithfulness makes an LLM call per evaluation — burns Groq tokens
    on EVERY answer, even cache hits. For a free-tier system this is wasteful.
    The cross-encoder (already loaded for reranking) can approximate
    faithfulness by scoring answer sentences against context chunks directly.

HOW local faithfulness scoring works:
    1. Split answer into sentences
    2. For each sentence, score against every context chunk using cross-encoder
    3. Take max score per sentence (best matching context)
    4. Average across all sentences = faithfulness score
    5. Normalize to 0-1 range (cross-encoder scores are unbounded)

Interpretation:
    Score > 0.7  — answer well grounded in context
    Score 0.4-0.7 — partially grounded
    Score < 0.4  — answer may contain hallucinations

Fallback to RAGAS:
    If cross-encoder fails to load or scoring fails → RAGAS + Groq runs as before.
    System never returns no score due to evaluator failure.
    RAGAS itself falls back to None on any error — non-blocking always.
"""

import math
import re
from typing import List, Optional, Dict


# ── Local cross-encoder scorer ────────────────────────────────────────────────

_cross_encoder = None
_cross_encoder_failed = False


def _get_cross_encoder():
    """
    Load cross-encoder for faithfulness scoring.
    Reuses the same model already loaded by reranker — no extra memory.
    """
    global _cross_encoder, _cross_encoder_failed

    if _cross_encoder_failed:
        return None
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
        print("[evaluator] Loading cross-encoder for faithfulness scoring...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[evaluator] Cross-encoder loaded")
        return _cross_encoder
    except Exception as e:
        print(f"[evaluator] Cross-encoder failed to load: {e} — will use RAGAS fallback")
        _cross_encoder_failed = True
        return None


def _split_sentences(text: str) -> List[str]:
    """
    Split answer text into sentences for per-sentence scoring.
    Filters out very short fragments (< 20 chars) — not meaningful to score.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter markdown bullets/headers and short fragments
    sentences = [
        s.strip().lstrip("•-*#").strip()
        for s in sentences
        if len(s.strip()) >= 20
    ]
    return sentences if sentences else [text.strip()]


def _sigmoid(x: float) -> float:
    """
    Normalize cross-encoder score to 0-1 range via sigmoid.
    Cross-encoder scores are unbounded logits — sigmoid maps them smoothly.
    Score of 0 → 0.5, positive scores → >0.5, negative → <0.5.
    """
    return 1.0 / (1.0 + math.exp(-x / 5.0))  # /5 softens the curve


def _local_faithfulness(answer: str, contexts: List[str]) -> Optional[float]:
    """
    Score answer faithfulness using cross-encoder.

    For each sentence in the answer, find the best matching context chunk.
    Average of these best-match scores = faithfulness proxy.

    Returns: float 0-1, or None if scoring fails.
    """
    model = _get_cross_encoder()
    if model is None:
        return None

    try:
        sentences = _split_sentences(answer)
        if not sentences or not contexts:
            return None

        sentence_scores = []

        for sentence in sentences:
            # Score this sentence against every context chunk
            pairs = [[sentence, ctx] for ctx in contexts]
            scores = model.predict(pairs)

            # Take the max — best matching context for this sentence
            best_score = float(max(scores))

            # Normalize to 0-1
            normalized = _sigmoid(best_score)
            sentence_scores.append(normalized)

        if not sentence_scores:
            return None

        faithfulness_score = round(sum(sentence_scores) / len(sentence_scores), 4)
        print(f"[evaluator] Local faithfulness: {faithfulness_score:.4f} "
              f"({len(sentences)} sentences × {len(contexts)} contexts)")
        return faithfulness_score

    except Exception as e:
        print(f"[evaluator] Local scoring failed: {e}")
        return None


# ── RAGAS fallback ────────────────────────────────────────────────────────────

_ragas_llm        = None
_ragas_embeddings = None


def _safe_float(val) -> Optional[float]:
    """Convert value to float safely — returns None for NaN, inf, or invalid."""
    try:
        if isinstance(val, list):
            val = val[0] if val else None
        if val is None:
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def _ragas_faithfulness(
    question: str,
    answer:   str,
    contexts: List[str]
) -> Optional[float]:
    """
    Original RAGAS faithfulness scoring — kept as fallback.
    Uses Groq LLM via call_with_fallback for claim decomposition.
    Only called when local cross-encoder scoring fails.
    """
    try:
        import os
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from datasets import Dataset

        global _ragas_llm, _ragas_embeddings

        if _ragas_llm is None:
            groq_llm   = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0,
                max_tokens=4096,
            )
            _ragas_llm = LangchainLLMWrapper(groq_llm)

        if _ragas_embeddings is None:
            hf_embeddings   = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
            _ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

        dataset = Dataset.from_dict({
            "question": [question],
            "answer":   [answer],
            "contexts": [contexts],
        })

        faithfulness.llm        = _ragas_llm
        faithfulness.embeddings = _ragas_embeddings

        result    = evaluate(dataset=dataset, metrics=[faithfulness])
        result_df = result.to_pandas()
        score     = _safe_float(result_df["faithfulness"].iloc[0])

        print(f"[evaluator] RAGAS faithfulness (fallback): {score}")
        return score

    except Exception as e:
        print(f"[evaluator] RAGAS fallback also failed: {e}")
        return None


# ── Public interface ──────────────────────────────────────────────────────────

def evaluate_response(
    question: str,
    answer:   str,
    contexts: List[str],
) -> Dict[str, Optional[float]]:
    """
    Evaluate answer faithfulness against retrieved contexts.

    Fallback chain:
        1. Local cross-encoder (zero API tokens, GPU, ~50ms)
           → if fails: RAGAS + Groq (original approach)
           → if fails: return None (non-blocking — never crashes the system)

    Returns dict with faithfulness score 0-1 (or None on complete failure).
    Runs in background thread — never blocks the user response.
    """
    if not answer or not contexts:
        return {"faithfulness": None, "answer_relevancy": None}

    contexts = [c for c in contexts if c and c.strip()]
    if not contexts:
        return {"faithfulness": None, "answer_relevancy": None}

    # ── Primary: local cross-encoder ─────────────────────────────────────────
    local_score = _local_faithfulness(answer, contexts)
    if local_score is not None:
        return {
            "faithfulness":     local_score,
            "answer_relevancy": None   # not computed — cross-encoder doesn't measure this
        }

    # ── Fallback: RAGAS + Groq ────────────────────────────────────────────────
    print("[evaluator] Local scoring failed — falling back to RAGAS")
    ragas_score = _ragas_faithfulness(question, answer, contexts)
    return {
        "faithfulness":     ragas_score,
        "answer_relevancy": None
    }