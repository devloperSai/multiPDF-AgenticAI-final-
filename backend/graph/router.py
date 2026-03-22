"""
Intent Router — Production-grade, three-tier classification
------------------------------------------------------------

Tier 1 — Zero-shot NLI classifier (PRIMARY)
    Model : cross-encoder/nli-MiniLM2-L6-H768  (~90MB)
    Device: GPU (GTX 1650) — ~5ms per query
    Why   : Understands meaning, not characters. Zero false positives.
            No API calls, no tokens burned, no keyword maintenance.
    Load  : Once at startup, stays in GPU memory alongside reranker.

Tier 2 — Two-tier keyword matcher (FALLBACK if model fails to load)
    Tier 2a: Single keywords, fuzzy matched (catches typos)
    Tier 2b: Anchor phrases, exact match only (no false positives)
    Why   : Previous production-fixed keyword logic kept as safety net.
            If NLI model can't load (no internet, disk issue), system
            still works — just falls back to keyword matching.

Tier 3 — Default factual
    If everything fails → return "factual" (safest default).
    System never crashes due to intent classification failure.

Fallback chain:
    NLI model → keyword matcher → "factual"
    No LLM calls at any stage.
"""

from typing import Literal

# NLI uses descriptive labels — the model matches these against the question meaning
# "factual" alone is weak; "specific fact or detail" is what the model understands
INTENT_LABELS    = ["factual", "summary", "comparison", "out_of_scope"]
INTENT_LABELS_T  = Literal["factual", "summary", "comparison", "out_of_scope"]

# Human-readable descriptions for NLI hypothesis matching
# These replace the raw label names in the hypothesis template
# Better descriptions → higher confidence scores → fewer keyword fallbacks
NLI_LABEL_DESCRIPTIONS = {
    "factual":      "specific fact, detail, value, name, date, definition or explanation about one thing",
    "summary":      "overall summary, overview, or complete description of the entire document",
    "comparison":   "explicit comparison or contrast between two or more different documents",
    "out_of_scope": "topic completely unrelated to any document, like weather, sports or cooking",
}

# Confidence threshold for NLI model
# If top label scores below this → fall through to keyword matcher
# 0.5 = model must be at least moderately confident
from config import cfg
NLI_CONFIDENCE_THRESHOLD = cfg.NLI_CONFIDENCE_THRESHOLD

# ── NLI model — loaded once, reused for all requests ─────────────────────────
_nli_classifier = None
_nli_load_failed = False  # if True, skip NLI and go straight to keywords


def _get_nli_classifier():
    """
    Load zero-shot NLI classifier on GPU.
    Loaded once at first call, cached globally.
    Sets _nli_load_failed = True on any error so we never retry a broken load.
    """
    global _nli_classifier, _nli_load_failed

    if _nli_load_failed:
        return None

    if _nli_classifier is not None:
        return _nli_classifier

    try:
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"

        print(f"[router] Loading NLI classifier on {device_name}...")
        _nli_classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768",
            device=device,
        )
        print(f"[router] NLI classifier loaded on {device_name}")
        return _nli_classifier

    except Exception as e:
        print(f"[router] NLI model failed to load: {e} — falling back to keyword matcher")
        _nli_load_failed = True
        return None


def _classify_with_nli(question: str) -> str | None:
    """
    Tier 1: Zero-shot NLI classification.

    Returns intent string if confident, None if below threshold.
    None triggers fallback to keyword matcher — not an error.

    Hypothesis template explains what each label means in plain English
    so the NLI model can reason about it correctly.
    """
    classifier = _get_nli_classifier()
    if classifier is None:
        return None

    try:
        # Hypothesis template — tells NLI what each label means
        # "This question is asking for a {}" → model checks which label fits best
        # Use descriptive labels instead of raw intent names
        # Maps "factual" → "specific fact, detail, value, name, date, or explanation"
        # NLI model understands these descriptions much better than raw labels
        label_descriptions = [NLI_LABEL_DESCRIPTIONS[l] for l in INTENT_LABELS]

        result = classifier(
            question,
            candidate_labels=label_descriptions,
            hypothesis_template="This question is asking for a {}.",
            multi_label=False,
        )

        # Map description back to intent label
        top_desc  = result["labels"][0]
        top_label = next(
            k for k, v in NLI_LABEL_DESCRIPTIONS.items() if v == top_desc
        )

        top_score = result["scores"][0]

        print(f"[router] NLI → {top_label} (confidence={top_score:.3f})")

        # Only trust the result if model is sufficiently confident
        if top_score >= NLI_CONFIDENCE_THRESHOLD:
            return top_label

        # Low confidence — let keyword matcher handle it
        print(f"[router] NLI confidence {top_score:.3f} < {NLI_CONFIDENCE_THRESHOLD} — deferring to keywords")
        return None

    except Exception as e:
        print(f"[router] NLI inference failed: {e} — falling back to keyword matcher")
        return None


# ── Tier 2: Keyword matcher — fallback if NLI unavailable or low confidence ───

from rapidfuzz import fuzz

FUZZY_THRESHOLD = cfg.FUZZY_MATCH_THRESHOLD

SINGLE_KEYWORDS = {
    "summary": [
        "summarize", "summarise", "summerize", "summerise",
        "summery", "sumary", "sumarize",
        "overview", "highlights", "highlight",
        "synopsis", "tldr", "gist", "crux", "essence",
        "briefly", "summarizing",
    ],
    "comparison": [
        "compare", "contrast", "comparison", "comparative",
        "versus", "distinguish",
        "similarities", "similarity", "differences",
    ],
    "out_of_scope": [
        "weather", "forecast", "recipe",
        "joke", "sing", "poem",
    ],
}

ANCHOR_PHRASES = {
    "summary": [
        "what is this document about", "what does this document cover",
        "what is this paper about",    "what does this paper cover",
        "what is this pdf about",      "what does this pdf cover",
        "what is this about",          "what does this cover",
        "what is the document about",  "what is the paper about",
        "what is the pdf about",       "what does the document cover",
        "tell me about this document", "tell me about this paper",
        "what does this file contain", "what is this file about",
        "main points",    "key points",
        "key takeaways",  "main takeaways",
        "key findings",   "main findings",
        "key ideas",      "main ideas",
        "key highlights",
        "give me a summary", "give me summary", "give summary",
        "write a summary",   "provide a summary", "provide summary",
        "brief summary",     "short summary",     "quick summary",
        "in a nutshell",     "give me the gist",
        "explain this document", "explain this pdf", "explain this paper",
        "describe this document","describe this pdf",
        "tell me about this document", "tell me about this pdf",
        "summarize first pdf",  "summarize the first pdf",
        "summarize second pdf", "summarize the second pdf",
        "summarize all",        "summarize everything",
    ],
    "comparison": [
        "compare and contrast",
        "difference between",       "differences between",
        "what is the difference between",
        "what are the differences between",
        "how are they different",   "how do they differ",
        "distinction between",
        "what are the similarities","how are they similar",
        "what do they have in common",
        "common between",           "compared to",    "compared with",
        "both documents",           "both pdfs",      "both papers",
        "across documents",         "across pdfs",
        "between the two",          "between both",
        "which is better",          "which is worse", "which one is better",
    ],
    "out_of_scope": [
        "who is the president",  "what is the capital of",
        "colour of sky",         "color of sky",
        "how old is",            "when was born",
        "sports score",          "cricket score",   "football score",
        "how to cook",           "how to make food",
        "tell me a joke",        "make me laugh",
        "write a song",          "tell me a story", "bedtime story",
        "what is 2 plus 2",      "solve this equation",
        "capital of india",      "capital of usa",
        "what time is it",       "what is today's date",
        "what day is it",        "temperature outside",
        "will it rain",          "climate today",
    ],
}


def _match_single_keyword(query: str, intent: str) -> str | None:
    for keyword in SINGLE_KEYWORDS.get(intent, []):
        if len(keyword) <= 4:
            if keyword in query.split():
                return keyword
        else:
            if fuzz.partial_ratio(keyword, query) >= FUZZY_THRESHOLD:
                return keyword
    return None


def _match_anchor_phrase(query: str, intent: str) -> str | None:
    for phrase in ANCHOR_PHRASES.get(intent, []):
        if phrase in query:
            return phrase
    return None


def _classify_with_keywords(question: str) -> str:
    """
    Tier 2: Two-tier keyword classification.
    Used when NLI model is unavailable or low confidence.
    Returns intent string — always returns something (defaults to factual).
    """
    q = question.lower().strip()

    # out_of_scope first — fastest exit
    if _match_single_keyword(q, "out_of_scope") or _match_anchor_phrase(q, "out_of_scope"):
        matched = _match_single_keyword(q, "out_of_scope") or _match_anchor_phrase(q, "out_of_scope")
        print(f"[router] keyword out_of_scope matched: '{matched}'")
        return "out_of_scope"

    matched = _match_single_keyword(q, "summary") or _match_anchor_phrase(q, "summary")
    if matched:
        print(f"[router] keyword summary matched: '{matched}'")
        return "summary"

    matched = _match_single_keyword(q, "comparison") or _match_anchor_phrase(q, "comparison")
    if matched:
        print(f"[router] keyword comparison matched: '{matched}'")
        return "comparison"

    return "factual"


# ── Public interface ──────────────────────────────────────────────────────────

def classify_intent(question: str, pdf_count: int = 1) -> str:
    """
    Classify question intent.
    Returns: factual | summary | comparison | out_of_scope

    Fallback chain (no LLM calls at any stage):
        1. NLI model (local GPU, ~5ms, understands meaning)
           → if confident: return result
           → if low confidence: fall through to keywords
           → if model unavailable: fall through to keywords
        2. Keyword matcher (two-tier, production-fixed)
           → always returns a result
        3. Default "factual" (implicit — keyword matcher always returns something)

    pdf_count guard:
        Comparison requires >= 2 PDFs to make sense.
        If only 1 PDF in session → comparison is impossible → force factual.
        Prevents NLI false positives on factual questions with multiple nouns.
    """
    # Tier 1 — NLI model
    nli_result = _classify_with_nli(question)
    if nli_result is not None:
        # ── Single PDF guard — comparison impossible with 1 PDF ──────────────
        # Only overrides comparison → factual.
        # out_of_scope and summary are never overridden — they are always valid.
        if nli_result == "comparison" and pdf_count < 2:
            print(f"[router] NLI → comparison but only {pdf_count} PDF — overriding to factual")
            # Run keyword check — question might actually be out_of_scope
            keyword_check = _classify_with_keywords(question)
            if keyword_check == "out_of_scope":
                print(f"[router] Keyword check → out_of_scope — using that instead")
                return "out_of_scope"
            return "factual"
        print(f"[router] Intent detected: {nli_result} (via NLI)")
        return nli_result

    # Tier 2 — keyword matcher
    keyword_result = _classify_with_keywords(question)

    # Apply same guard to keyword result
    if keyword_result == "comparison" and pdf_count < 2:
        print(f"[router] keyword → comparison but only {pdf_count} PDF — overriding to factual")
        return "factual"

    print(f"[router] Intent detected: {keyword_result} (via keywords)")
    return keyword_result


def warm_up():
    """
    Pre-load NLI model at server startup so first request isn't slow.
    Call this from main.py on_startup event.
    """
    print("[router] Warming up NLI classifier...")
    _get_nli_classifier()
    if _nli_classifier is not None:
        # Run one dummy inference to initialize CUDA kernels
        try:
            _nli_classifier(
                "test question",
                candidate_labels=INTENT_LABELS,
                hypothesis_template="This question is asking for a {}.",
            )
            print("[router] NLI classifier warm-up complete")
        except Exception as e:
            print(f"[router] Warm-up inference failed: {e}")