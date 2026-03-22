"""
core/classifier.py

3-signal document classification:
  Signal 1 — Filename pattern matching  (weight: 0.40)
  Signal 2 — Embedding cosine similarity (weight: 0.40)
  Signal 3 — Keyword rule-based         (weight: 0.20)

Weighted combination corrects for cases where one signal is weak or wrong.
Example: research paper about legal AI
  - keywords say "legal" (wrong)
  - filename says "research" (right)
  - embeddings say "research" (right)
  - combined → "research" ✓
"""

import re
import os
from typing import Dict

# ── Signal 1: Filename patterns ───────────────────────────────────────────────

FILENAME_PATTERNS = {
    "legal": [
        "nda", "agreement", "contract", "legal", "terms", "conditions",
        "policy", "compliance", "disclosure", "memorandum", "affidavit",
        "deed", "clause", "mou", "settlement", "license", "licence",
        "privacy", "gdpr", "consent", "liability", "indemnity", "waiver",
        "confidential", "intellectual", "property", "trademark", "patent",
        "copyright", "employment", "service", "vendor", "consulting"
    ],
    "research": [
        "paper", "research", "study", "analysis", "survey", "arxiv",
        "journal", "thesis", "dissertation", "review", "proceedings",
        "abstract", "preprint", "publication", "conference", "workshop",
        "findings", "report", "experiment", "evaluation", "benchmark"
    ],
    "financial": [
        "financial", "finance", "revenue", "budget", "invoice", "balance",
        "quarterly", "annual", "profit", "loss", "tax", "audit", "statement",
        "earnings", "fiscal", "accounting", "ledger", "payroll", "expense",
        "forecast", "valuation", "portfolio", "investment", "fund"
    ],
    "general": [
        "guide", "manual", "handbook", "tutorial", "overview", "summary",
        "brochure", "catalog", "catalogue", "newsletter", "presentation",
        "slides", "info", "faq", "readme", "wiki", "tour", "darshan"
    ]
}


def _classify_from_filename(filename: str) -> Dict[str, float]:
    """
    Score doc types based on filename tokens.
    Returns normalized scores summing to 1.0.
    """
    if not filename:
        return {"legal": 0.25, "research": 0.25, "financial": 0.25, "general": 0.25}

    # Normalize: lowercase, split on _ - . spaces digits
    name = os.path.splitext(filename.lower())[0]
    name = re.sub(r'[_\-\.\s\d]+', ' ', name)
    tokens = set(name.split())

    scores = {"legal": 0.0, "research": 0.0, "financial": 0.0, "general": 0.0}

    for doc_type, patterns in FILENAME_PATTERNS.items():
        for token in tokens:
            if token in patterns:
                scores[doc_type] += 1.0
            # Partial match for compound words like "nondisclosure"
            for pattern in patterns:
                if len(pattern) > 4 and pattern in name:
                    scores[doc_type] += 0.5

    total = sum(scores.values())
    if total == 0:
        # No filename signal — return uniform distribution
        return {"legal": 0.25, "research": 0.25, "financial": 0.25, "general": 0.25}

    # Normalize
    return {k: v / total for k, v in scores.items()}


# ── Signal 2: Embedding similarity ───────────────────────────────────────────

# Reference descriptions — what each doc type "looks like" semantically
REFERENCE_DESCRIPTIONS = {
    "legal": (
        "This is a legal document containing contracts, agreements, clauses, "
        "obligations, rights, terms and conditions, liability, jurisdiction, "
        "parties, warranties, indemnification, confidentiality, governing law, "
        "representations, breach, remedy, enforcement, arbitration, executed."
    ),
    "research": (
        "This is a research paper containing abstract, introduction, methodology, "
        "literature review, findings, conclusions, citations, experiments, "
        "hypothesis, results, analysis, dataset, evaluation, benchmark, "
        "proposed method, algorithm, neural network, training, model performance."
    ),
    "financial": (
        "This is a financial document containing revenue, profit, loss, balance sheet, "
        "income statement, cash flow, assets, liabilities, equity, earnings per share, "
        "quarterly results, annual report, fiscal year, budget, audit, depreciation, "
        "gross margin, net income, operating expenses, EBITDA, valuation."
    ),
    "general": (
        "This is a general document containing information, descriptions, guide, "
        "overview, summary, topics, places, travel, food, culture, history, "
        "tourism, attractions, restaurants, locations, activities, events."
    )
}

# Cache: reference embeddings computed once, reused for all classifications
_reference_embeddings: Dict[str, list] = {}


def _get_reference_embeddings() -> Dict[str, list]:
    """
    Compute and cache reference embeddings for each doc type.
    Uses the same BAAI/bge-base-en-v1.5 model already loaded for chunks.
    """
    global _reference_embeddings
    if _reference_embeddings:
        return _reference_embeddings

    try:
        from core.embedder import embed_query
        print("[classifier] Computing reference embeddings...")
        for doc_type, description in REFERENCE_DESCRIPTIONS.items():
            _reference_embeddings[doc_type] = embed_query(description)
        print("[classifier] Reference embeddings ready")
    except Exception as e:
        print(f"[classifier] Embedding failed: {e}")
        _reference_embeddings = {}

    return _reference_embeddings


def _cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b:
        return 0.0
    try:
        import math
        dot    = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception:
        return 0.0


def _classify_from_embedding(text: str) -> Dict[str, float]:
    """
    Score doc types by cosine similarity between doc embedding
    and reference description embeddings.
    Returns normalized scores.
    """
    try:
        from core.embedder import embed_query

        ref_embeddings = _get_reference_embeddings()
        if not ref_embeddings:
            return {"legal": 0.25, "research": 0.25, "financial": 0.25, "general": 0.25}

        # Use first 512 tokens worth of text for classification
        excerpt = text[:2000].strip()
        if not excerpt:
            return {"legal": 0.25, "research": 0.25, "financial": 0.25, "general": 0.25}

        doc_embedding = embed_query(excerpt)

        scores = {}
        for doc_type, ref_emb in ref_embeddings.items():
            scores[doc_type] = max(0.0, _cosine_similarity(doc_embedding, ref_emb))

        total = sum(scores.values())
        if total == 0:
            return {"legal": 0.25, "research": 0.25, "financial": 0.25, "general": 0.25}

        return {k: v / total for k, v in scores.items()}

    except Exception as e:
        print(f"[classifier] Embedding classification failed: {e}")
        return {"legal": 0.25, "research": 0.25, "financial": 0.25, "general": 0.25}


# ── Signal 3: Keyword rule-based ─────────────────────────────────────────────

KEYWORD_SETS = {
    "research": [
        "abstract", "introduction", "methodology", "conclusion", "references",
        "arxiv", "hypothesis", "experiment", "algorithm", "literature review",
        "findings", "proposed method", "dataset", "evaluation", "benchmark",
        "neural", "training", "model", "citation", "figure", "table", "appendix"
    ],
    "legal": [
        "agreement", "contract", "whereas", "hereby", "jurisdiction",
        "plaintiff", "defendant", "clause", "liability", "court",
        "consultant", "indemnification", "indemnify", "reimbursement",
        "termination", "confidentiality", "confidential information",
        "intellectual property", "governing law", "arbitration",
        "representations", "warranties", "obligations", "executed",
        "effective date", "parties", "scope of work", "non-disclosure",
        "breach", "remedy", "enforcement", "severability", "entire agreement"
    ],
    "financial": [
        "revenue", "balance sheet", "profit", "loss", "fiscal",
        "earnings", "invoice", "budget", "cash flow", "audit",
        "assets", "liabilities", "equity", "income statement",
        "quarterly", "annual report", "ebitda", "gross margin",
        "net income", "operating expenses", "depreciation"
    ],
    "general": [
        "welcome", "guide", "overview", "introduction to", "about us",
        "located", "visit", "tourism", "attraction", "restaurant",
        "hotel", "travel", "city", "culture", "history", "food"
    ]
}


def _classify_from_keywords(text: str) -> Dict[str, float]:
    """
    Score doc types by keyword frequency.
    Returns normalized scores.
    """
    excerpt = text[:3000].lower()
    scores  = {doc_type: 0.0 for doc_type in KEYWORD_SETS}

    for doc_type, keywords in KEYWORD_SETS.items():
        for kw in keywords:
            if kw in excerpt:
                scores[doc_type] += 1.0

    total = sum(scores.values())
    if total == 0:
        return {"legal": 0.25, "research": 0.25, "financial": 0.25, "general": 0.25}

    return {k: v / total for k, v in scores.items()}


# ── Main classifier ───────────────────────────────────────────────────────────

# Weights for each signal
WEIGHTS = {
    "filename":  0.40,
    "embedding": 0.40,
    "keywords":  0.20,
}


def classify_document(text: str, filename: str = "") -> str:
    """
    Classify document type using 3 combined signals:
      - Filename pattern matching (0.40)
      - Embedding cosine similarity (0.40)
      - Keyword rule-based (0.20)

    Returns: "legal" | "research" | "financial" | "general"
    """
    doc_types = ["legal", "research", "financial", "general"]

    # Run all 3 signals
    filename_scores  = _classify_from_filename(filename)
    embedding_scores = _classify_from_embedding(text)
    keyword_scores   = _classify_from_keywords(text)

    # Weighted combination
    combined = {}
    for dt in doc_types:
        combined[dt] = (
            WEIGHTS["filename"]  * filename_scores.get(dt, 0.25) +
            WEIGHTS["embedding"] * embedding_scores.get(dt, 0.25) +
            WEIGHTS["keywords"]  * keyword_scores.get(dt, 0.25)
        )

    best       = max(combined, key=combined.get)
    confidence = round(combined[best], 4)

    print(f"[classifier] Filename:  {_fmt(filename_scores)}")
    print(f"[classifier] Embedding: {_fmt(embedding_scores)}")
    print(f"[classifier] Keywords:  {_fmt(keyword_scores)}")
    print(f"[classifier] Combined → '{best}' (score: {confidence})")

    return best


def _fmt(scores: Dict[str, float]) -> str:
    """Format scores dict for logging."""
    return " | ".join(f"{k}:{round(v,2)}" for k, v in scores.items())