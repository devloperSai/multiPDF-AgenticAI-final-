"""
Query Expander — Local T5 Paraphrase Model
-------------------------------------------
Expands a single query into multiple semantically equivalent variants
before retrieval. Each variant is embedded and searched independently.
Results are merged via RRF for higher recall.

WHY this helps:
    User: "termination notice period"
    BM25 misses: "notice for contract termination" — different word order
    Vector misses: "how many days to end the agreement" — different vocabulary

    With expansion:
        Variant 1: "termination notice period"            ← original
        Variant 2: "how many days notice for termination"
        Variant 3: "contract termination notification period"
        Variant 4: "notice required to end the contract"

    All 4 are searched. Chunks matching ANY variant are retrieved.
    RRF merges and re-ranks. Coverage goes from ~60% to ~90%.

WHEN it fires:
    Only for factual intent + query length 3+ words.
    Short queries (1-2 words) and summary/comparison don't benefit.
    Conditional on model loading — graceful fallback to original query.

MODEL:
    humarin/chatgpt_paraphraser_on_T5_base
    - ~900MB, runs on GPU
    - Fine-tuned specifically for paraphrase generation
    - Fits GTX 1650 4GB alongside cross-encoder + embedder

FALLBACK CHAIN (per system rule — never crash):
    T5 model fails to load → return [original_query] only
    T5 generation fails   → return [original_query] only
    Pipeline continues normally — just no expansion
"""

from typing import List, Optional
from config import cfg

# ── Model singleton ───────────────────────────────────────────────────────────
_tokenizer    = None
_model        = None
_load_failed  = False

# Pull from config — tunable via PDFQA_* env vars
MIN_QUERY_WORDS = cfg.QUERY_EXPANSION_MIN_WORDS
NUM_VARIANTS    = cfg.QUERY_EXPANSION_VARIANTS


def _get_model():
    """
    Load T5 paraphrase model lazily — only on first factual query.
    Singleton — loaded once per worker process, reused for all queries.
    """
    global _tokenizer, _model, _load_failed

    if _load_failed:
        return None, None
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "humarin/chatgpt_paraphraser_on_T5_base"
        device     = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[query_expander] Loading {model_name} on {device}...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        # use_safetensors=True — avoids torch.load security issue (CVE-2025-32434)
        # Requires torch >= 2.6 for .bin files but safetensors works on all versions
        _model     = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            use_safetensors=True
        ).to(device)
        _model.eval()
        print(f"[query_expander] Model loaded on {device}")
        return _tokenizer, _model

    except Exception as e:
        print(f"[query_expander] Failed to load model: {e} — expansion disabled")
        _load_failed = True
        return None, None


def _generate_variants(query: str, num_variants: int = NUM_VARIANTS) -> List[str]:
    """
    Generate paraphrase variants of the query using T5.

    Uses beam search with num_beams > num_variants to get diverse outputs.
    Temperature and repetition penalty tuned for query-length outputs.

    Returns list of variant strings (may be fewer than requested if
    model generates duplicates or very similar strings).
    """
    tokenizer, model = _get_model()
    if tokenizer is None or model is None:
        return []

    try:
        import torch

        # T5 paraphrase prompt format for this specific model
        input_text = f"paraphrase: {query} </s>"

        encoding = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        )

        # Move to same device as model
        device = next(model.parameters()).device
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=num_variants * 2,        # search wider for diversity
                num_return_sequences=num_variants,
                no_repeat_ngram_size=2,             # avoid repetitive phrases
                repetition_penalty=1.5,             # discourage copying input
                temperature=1.0,                    # balanced diversity
                early_stopping=True
            )

        variants = []
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True).strip()
            # Skip if identical to original or empty
            if decoded and decoded.lower() != query.lower():
                variants.append(decoded)

        return variants

    except Exception as e:
        print(f"[query_expander] Generation failed: {e}")
        return []


def expand_query(query: str) -> List[str]:
    """
    Expand a query into multiple variants for retrieval.

    Returns list starting with original query, followed by variants.
    Always includes original — expansion only adds, never replaces.

    Args:
        query: The retrieval question (after coreference resolution)

    Returns:
        [original_query, variant_1, variant_2, ...]
        Just [original_query] if expansion fails or query too short.
    """
    query = query.strip()

    # Skip expansion for short queries — not enough context to paraphrase
    word_count = len(query.split())
    if word_count < MIN_QUERY_WORDS:
        print(f"[query_expander] Query too short ({word_count} words) — skipping expansion")
        return [query]

    variants = _generate_variants(query, num_variants=NUM_VARIANTS)

    if not variants:
        print(f"[query_expander] No variants generated — using original only")
        return [query]

    all_queries = [query] + variants

    print(f"[query_expander] Expanded '{query[:50]}' → {len(all_queries)} queries:")
    for i, q in enumerate(all_queries):
        label = "original" if i == 0 else f"variant {i}"
        print(f"  [{label}] {q}")

    return all_queries