from typing import List
from collections import Counter

# Model configurations per doc type
#
# provider field: which provider to use for this doc type
#   "groq"      — free, fast, default for all types
#   "openai"    — paid, optional (only activates if OPENAI_API_KEY in .env)
#   "anthropic" — paid, optional (only activates if ANTHROPIC_API_KEY in .env)
#   "together"  — free fallback, not used as primary
#
# Current: all doc types use Groq as primary.
# To route legal docs to Claude: change legal provider to "anthropic"
# The fallback chain still applies regardless of primary provider.
#
MODEL_CONFIGS = {
    "research": {
        "provider":    "groq",
        "model":       "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "max_tokens":  1500,
        "system_suffix": """You are analyzing a research paper.
Focus on: methodology, findings, conclusions, and citations.
Be precise and technical. Always distinguish between what the paper claims vs what is proven."""
    },
    "legal": {
        "provider":    "groq",                     # change to "anthropic" for Claude
        "model":       "llama-3.3-70b-versatile",
        "temperature": 0.0,                         # zero temp — legal needs deterministic answers
        "max_tokens":  1500,
        "system_suffix": """You are analyzing a legal document.
Focus on: exact clauses, obligations, rights, penalties, and jurisdictions.
Be literal and precise — never paraphrase legal terms loosely.
Always cite the specific clause or section when answering."""
    },
    "financial": {
        "provider":    "groq",                     # change to "openai" for GPT-4o
        "model":       "llama-3.3-70b-versatile",
        "temperature": 0.1,
        "max_tokens":  1500,
        "system_suffix": """You are analyzing a financial document.
Focus on: exact figures, percentages, dates, and financial terms.
Never approximate numbers — always use exact values from the document.
Clearly distinguish between revenue, profit, loss, and other financial metrics."""
    },
    "general": {
        "provider":    "groq",
        "model":       "llama-3.3-70b-versatile",
        "temperature": 0.3,
        "max_tokens":  1024,
        "system_suffix": """You are analyzing a general document.
Provide clear, accessible answers that are easy to understand."""
    }
}

DEFAULT_CONFIG = MODEL_CONFIGS["general"]

# Priority order — most restrictive wins when mixed doc types in session
PRIORITY = ["legal", "financial", "research", "general"]


def get_llm_config(doc_types: List[str]) -> dict:
    """
    Session-level config — used at route_node time for awareness.
    Still used as fallback if chunks have no doc_type metadata.
    Priority: legal > financial > research > general
    """
    for doc_type in PRIORITY:
        if doc_type in doc_types:
            config = MODEL_CONFIGS[doc_type].copy()
            config["doc_type"] = doc_type
            print(f"[model_router] Session-level config: {doc_type}")
            return config

    print(f"[model_router] No matching config, using default")
    return {**DEFAULT_CONFIG, "doc_type": "general"}


def get_llm_config_from_chunks(chunks: list) -> dict:
    """
    Refine 2 — Per-question LLM config based on retrieved chunk doc_types.

    Instead of using session-level doc_types, look at which chunks were
    actually retrieved and pick config based on their doc_type metadata.

    Logic:
    - Extract doc_type from each chunk's metadata
    - Find most common doc_type among retrieved chunks
    - If tie or mixed — most restrictive wins (legal > financial > research > general)
    - Falls back to general if no doc_type in metadata (old chunks before this refine)
    """
    if not chunks:
        print("[model_router] No chunks — using default config")
        return {**DEFAULT_CONFIG, "doc_type": "general"}

    doc_types_found = [
        c["metadata"].get("doc_type")
        for c in chunks
        if c.get("metadata", {}).get("doc_type")
    ]

    if not doc_types_found:
        print("[model_router] No doc_type in chunk metadata — using general (old chunks)")
        return {**DEFAULT_CONFIG, "doc_type": "general"}

    counts = Counter(doc_types_found)
    print(f"[model_router] Chunk doc_type distribution: {dict(counts)}")

    if len(counts) == 1:
        doc_type = list(counts.keys())[0]
        config = MODEL_CONFIGS.get(doc_type, DEFAULT_CONFIG).copy()
        config["doc_type"] = doc_type
        print(f"[model_router] Single doc_type from chunks: {doc_type}")
        return config

    present_types = list(counts.keys())
    for doc_type in PRIORITY:
        if doc_type in present_types:
            config = MODEL_CONFIGS[doc_type].copy()
            config["doc_type"] = doc_type
            print(f"[model_router] Mixed doc_types {present_types} — using most restrictive: {doc_type}")
            return config

    return {**DEFAULT_CONFIG, "doc_type": "general"}