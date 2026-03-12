"""
Refine 6 — Input Validation
-----------------------------
Validates user questions before they enter the pipeline.
Checks:
1. Length — min 3 chars, max 1000 chars
2. Not blank / whitespace only
3. Prompt injection patterns
4. Gibberish / symbol-only detection
"""

import re
from fastapi import HTTPException

MIN_LENGTH = 3
MAX_LENGTH = 1000

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"you\s+are\s+now\s+a\s+different",
    r"act\s+as\s+(if\s+you\s+are\s+)?a\s+different",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"you\s+are\s+no\s+longer",
    r"new\s+persona",
    r"system\s*:\s*you\s+are",
    r"<\s*system\s*>",
    r"\[system\]",
    r"###\s*instruction",
    r"override\s+(system|prompt|instructions)",
    r"bypass\s+(safety|filter|restriction)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"dan\s+mode",
]

_INJECTION_REGEX = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def _has_enough_real_words(text: str) -> bool:
    """
    Check if text contains at least one real word (3+ consecutive letters).
    Catches: "kkk", "jhvuyvyv", "[}{]]+", symbol soup.
    Does NOT block: "What is AI?", "CEO's salary", "Article 6".
    """
    return bool(re.search(r'[a-zA-Z]{3,}', text))


def _is_symbol_soup(text: str) -> bool:
    """
    Detect inputs that are mostly symbols/numbers with no real words.
    Threshold: >60% non-alpha characters AND no real word found.
    """
    stripped = text.strip()
    if not stripped:
        return False

    alpha_count = sum(1 for c in stripped if c.isalpha())
    alpha_ratio = alpha_count / len(stripped)

    # If less than 20% of chars are letters — almost certainly garbage
    if alpha_ratio < 0.2:
        return True

    return False


def validate_question(question: str):
    """
    Validate a user question before it enters the RAG pipeline.
    Raises HTTPException with specific, user-friendly messages.
    Returns silently if valid.
    """
    stripped = question.strip()

    # 1. Blank check
    if not stripped:
        raise HTTPException(
            status_code=400,
            detail="Please type a question before sending."
        )

    # 2. Minimum length
    if len(stripped) < MIN_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Your question is too short. Please ask a complete question (at least {MIN_LENGTH} characters)."
        )

    # 3. Maximum length
    if len(stripped) > MAX_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Your question is too long ({len(stripped)} characters). Please keep it under {MAX_LENGTH} characters."
        )

    # 4. Symbol soup — catches "[}{]]+" and similar
    if _is_symbol_soup(stripped):
        raise HTTPException(
            status_code=400,
            detail="Your question contains too many symbols. Please ask a clear question about your documents."
        )

    # 5. No real words — catches "kkk", "jhvuyvyv ihihviy hiy"
    if not _has_enough_real_words(stripped):
        raise HTTPException(
            status_code=400,
            detail="Please ask a proper question using real words."
        )

    # 6. Prompt injection
    for pattern in _INJECTION_REGEX:
        if pattern.search(stripped):
            raise HTTPException(
                status_code=400,
                detail="Invalid question format. Please ask a question about your documents."
            )