"""
config.py — Centralized System Configuration
---------------------------------------------
Single source of truth for ALL tunable constants in the system.
No more hunting through 10 files to change one threshold.

HOW TO USE:
    from config import cfg

    # Then use any value:
    cfg.RERANKER_THRESHOLD        # -8.0
    cfg.CHUNK_MAX_SIZE            # 1500
    cfg.SUMMARY_SPREAD_CHUNKS     # 12

HOW TO TUNE FOR DEPLOYMENT:
    All values can be overridden via environment variables.
    Format: same name as constant, prefixed with PDFQA_
    Example: PDFQA_RERANKER_THRESHOLD=-10.0 uvicorn main:app

    This means you can tune thresholds in production without
    touching code or redeploying — just change the env var.

SECTIONS:
    1. Models          — which models to load locally
    2. Retrieval       — search, HyDE, confidence thresholds
    3. Reranker        — cross-encoder scoring
    4. Chunking        — PDF splitting parameters
    5. Cache           — semantic cache settings
    6. Memory          — conversation summary settings
    7. Generation      — LLM generation settings
    8. Router          — intent classification thresholds
    9. Evaluation      — faithfulness scoring
    10. Storage        — file paths, PDF caps
    11. Providers      — LLM provider URLs and model names
"""

import os


def _env(key: str, default):
    """
    Read config value from environment variable if set, else use default.
    Env var name format: PDFQA_{key}
    Automatically casts to the same type as default.
    """
    env_key = f"PDFQA_{key}"
    val = os.getenv(env_key)
    if val is None:
        return default
    try:
        if isinstance(default, bool):
            return val.lower() in ("1", "true", "yes")
        return type(default)(val)
    except (ValueError, TypeError):
        print(f"[config] Warning: could not cast {env_key}={val!r} to {type(default).__name__} — using default {default!r}")
        return default


class Config:
    """
    All system constants in one place.
    Grouped by subsystem. Every value is tunable via env var.
    """

    # ── 1. Models ─────────────────────────────────────────────────────────────
    EMBED_MODEL         = _env("EMBED_MODEL",       "BAAI/bge-base-en-v1.5")
    EMBED_BATCH_SIZE    = _env("EMBED_BATCH_SIZE",  32)       # safe for GTX 1650 4GB VRAM
    RERANKER_MODEL      = _env("RERANKER_MODEL",    "cross-encoder/ms-marco-MiniLM-L-6-v2")
    NLI_MODEL           = _env("NLI_MODEL",         "cross-encoder/nli-MiniLM2-L6-H768")
    PRIMARY_LLM_MODEL   = _env("PRIMARY_LLM_MODEL", "llama-3.3-70b-versatile")

    # ── 2. Retrieval ──────────────────────────────────────────────────────────
    # HyDE: triggered when top RRF score below this threshold
    # RRF scores range 0.01-0.03 — 0.02 = middle of range
    HYDE_CONFIDENCE_THRESHOLD  = _env("HYDE_CONFIDENCE_THRESHOLD",  0.02)

    # Answer confidence gating: below this RRF score → return partial answer
    # instead of hallucinating. 0.013 = bottom quartile of RRF scores.
    CONFIDENCE_GATE_THRESHOLD  = _env("CONFIDENCE_GATE_THRESHOLD",  0.013)

    # Retrieval top_k per intent
    RETRIEVAL_TOP_K_FACTUAL    = _env("RETRIEVAL_TOP_K_FACTUAL",    6)
    RETRIEVAL_TOP_K_COMPARISON = _env("RETRIEVAL_TOP_K_COMPARISON", 10)

    # Summary spread retrieval: chunks sampled across entire document
    SUMMARY_SPREAD_CHUNKS      = _env("SUMMARY_SPREAD_CHUNKS",      12)

    # ── 3. Reranker ───────────────────────────────────────────────────────────
    # Cross-encoder score threshold for factual intent.
    # Calibrated on real data: irrelevant chunks score ~-11, relevant ~+3 to +15.
    # -8 sits safely between them. Summary/comparison intent skips this filter.
    RERANKER_THRESHOLD         = _env("RERANKER_THRESHOLD",         -9.0)   # lowered from -8 — general/travel docs need more breathing room

    # Reranker top_k for factual and comparison
    RERANKER_TOP_K             = _env("RERANKER_TOP_K",             6)

    # Chunk deduplication: Jaccard similarity threshold
    # 0.85 = chunks must differ by >=15% token overlap to both be kept
    CHUNK_DEDUP_THRESHOLD      = _env("CHUNK_DEDUP_THRESHOLD",      0.85)

    # ── 4. Chunking ───────────────────────────────────────────────────────────
    CHUNK_MIN_SIZE             = _env("CHUNK_MIN_SIZE",             100)   # ignore tiny fragments
    CHUNK_MAX_SIZE             = _env("CHUNK_MAX_SIZE",             1500)  # hard cap per chunk
    CHUNK_TARGET_SIZE          = _env("CHUNK_TARGET_SIZE",          800)   # ideal chunk size
    CHUNK_OVERLAP_CHARS        = _env("CHUNK_OVERLAP_CHARS",        150)   # overlap between chunks

    # ── 5. Cache ──────────────────────────────────────────────────────────────
    # Semantic cache: embedding similarity threshold for cache hit
    # 0.92 = questions must be 92% similar to share a cache entry
    CACHE_SIMILARITY_THRESHOLD = _env("CACHE_SIMILARITY_THRESHOLD", 0.92)
    CACHE_MAX_SIZE             = _env("CACHE_MAX_SIZE",             100)   # entries per session

    # ── 6. Memory ─────────────────────────────────────────────────────────────
    # Summary memory: how many recent messages to keep raw (not summarized)
    MEMORY_RAW_TAIL            = _env("MEMORY_RAW_TAIL",            3)

    # Summary memory: minimum messages before summarization kicks in
    # Below this count — inject all messages raw (summarizing is pointless)
    MEMORY_SUMMARY_THRESHOLD   = _env("MEMORY_SUMMARY_THRESHOLD",   6)

    # Summary memory: max tokens for the compressed summary paragraph
    MEMORY_SUMMARY_MAX_TOKENS  = _env("MEMORY_SUMMARY_MAX_TOKENS",  300)

    # ── 7. Generation ─────────────────────────────────────────────────────────
    # Max retries when LLM answer is insufficient
    MAX_RETRIES                = _env("MAX_RETRIES",                2)

    # HyDE generation: max tokens for hypothetical passage
    HYDE_MAX_TOKENS            = _env("HYDE_MAX_TOKENS",            200)
    HYDE_TEMPERATURE           = _env("HYDE_TEMPERATURE",           0.3)

    # ── 7b. Query Expansion ──────────────────────────────────────────────────────
    QUERY_EXPANSION_ENABLED    = _env("QUERY_EXPANSION_ENABLED",    True)
    QUERY_EXPANSION_VARIANTS   = _env("QUERY_EXPANSION_VARIANTS",   3)     # variants to generate
    QUERY_EXPANSION_MIN_WORDS  = _env("QUERY_EXPANSION_MIN_WORDS",  3)     # min words to trigger
    QUERY_EXPANSION_MODEL      = _env("QUERY_EXPANSION_MODEL",      "humarin/chatgpt_paraphraser_on_T5_base")

    # ── 8. Router ─────────────────────────────────────────────────────────────
    # NLI classifier: minimum confidence to trust the model
    # Below this → fall through to keyword matcher
    NLI_CONFIDENCE_THRESHOLD   = _env("NLI_CONFIDENCE_THRESHOLD",   0.50)

    # Keyword fuzzy match threshold (single keywords only)
    FUZZY_MATCH_THRESHOLD      = _env("FUZZY_MATCH_THRESHOLD",      85)

    # ── 9. Evaluation ─────────────────────────────────────────────────────────
    # Local faithfulness: sigmoid scale factor for score normalization
    # Lower = softer curve, higher = sharper
    EVAL_SIGMOID_SCALE         = _env("EVAL_SIGMOID_SCALE",         5.0)

    # Minimum sentence length to include in faithfulness scoring
    EVAL_MIN_SENTENCE_LEN      = _env("EVAL_MIN_SENTENCE_LEN",      20)

    # ── 10. Storage ───────────────────────────────────────────────────────────
    UPLOAD_DIR                 = _env("UPLOAD_DIR",                 "uploads")
    PDF_CAP                    = _env("PDF_CAP",                    100)   # global max PDFs
    PDF_EVICT_TO               = _env("PDF_EVICT_TO",               98)    # evict down to this

    # ── 11. Providers ─────────────────────────────────────────────────────────
    TOGETHER_BASE_URL          = _env("TOGETHER_BASE_URL",          "https://api.together.xyz/v1")
    TOGETHER_MODEL             = _env("TOGETHER_MODEL",             "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    OPENAI_MODEL               = _env("OPENAI_MODEL",               "gpt-4o-mini")
    ANTHROPIC_MODEL            = _env("ANTHROPIC_MODEL",            "claude-haiku-4-5-20251001")


# ── Singleton instance ────────────────────────────────────────────────────────
# Import this everywhere: from config import cfg
cfg = Config()


# ── Startup log ───────────────────────────────────────────────────────────────
def log_config():
    """Print active config at startup so values are visible in logs."""
    print("[config] Active configuration:")
    print(f"  Embed model:          {cfg.EMBED_MODEL}")
    print(f"  Reranker threshold:   {cfg.RERANKER_THRESHOLD}")
    print(f"  HyDE threshold:       {cfg.HYDE_CONFIDENCE_THRESHOLD}")
    print(f"  Cache similarity:     {cfg.CACHE_SIMILARITY_THRESHOLD}")
    print(f"  NLI confidence:       {cfg.NLI_CONFIDENCE_THRESHOLD}")
    print(f"  Chunk max size:       {cfg.CHUNK_MAX_SIZE}")
    print(f"  Summary spread:       {cfg.SUMMARY_SPREAD_CHUNKS} chunks")
    print(f"  PDF cap:              {cfg.PDF_CAP}")