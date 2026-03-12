import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
CANDIDATE_LABELS = ["research paper", "legal document", "financial report", "general document"]

LABEL_MAP = {
    "research paper": "research",
    "legal document": "legal",
    "financial report": "financial",
    "general document": "general"
}


def classify_document(text: str) -> str:
    """
    Classify document using HuggingFace zero-shot classification.
    Uses facebook/bart-large-mnli — free, no billing required.
    Falls back to rule-based if API fails.
    """
    try:
        excerpt = text[:1000].strip()
        if not excerpt:
            return "general"

        headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
        payload = {
            "inputs": excerpt,
            "parameters": {"candidate_labels": CANDIDATE_LABELS}
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 503:
            # Model is loading — wait and retry once
            import time
            print("[classifier] Model loading, retrying in 20s...")
            time.sleep(20)
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

        response.raise_for_status()
        result = response.json()

        top_label = result["labels"][0]
        top_score = result["scores"][0]
        doc_type = LABEL_MAP.get(top_label, "general")

        print(f"[classifier] Classified as '{doc_type}' (confidence: {round(top_score, 3)})")
        return doc_type

    except Exception as e:
        print(f"[classifier] HF API failed: {e} — using rule-based fallback")
        return _rule_based_classify(text)


def _rule_based_classify(text: str) -> str:
    """
    Fallback if HuggingFace API is unavailable.
    Threshold lowered to 1 — any single strong keyword match is enough.
    Keywords expanded to cover consulting agreements, NDAs, service contracts.
    """
    excerpt = text[:3000].lower()

    scores = {
        "research": sum(1 for k in [
            "abstract", "introduction", "methodology", "conclusion",
            "references", "arxiv", "hypothesis", "experiment", "algorithm",
            "literature review", "findings", "proposed method", "dataset",
            "evaluation", "benchmark", "neural", "training", "model"
        ] if k in excerpt),

        "legal": sum(1 for k in [
            # Classic legal
            "agreement", "contract", "whereas", "hereby", "jurisdiction",
            "plaintiff", "defendant", "clause", "liability", "court",
            # Contract/consulting specific
            "consultant", "consulting", "independent contractor",
            "indemnification", "indemnify", "reimbursement", "compensation",
            "termination", "confidentiality", "confidential information",
            "intellectual property", "governing law", "arbitration",
            "representations", "warranties", "obligations", "executed",
            "effective date", "party", "parties", "scope of work",
            "services rendered", "non-disclosure", "proprietary",
            "breach", "remedy", "enforcement", "severability",
            "entire agreement", "amendment", "waiver", "notice"
        ] if k in excerpt),

        "financial": sum(1 for k in [
            "revenue", "balance sheet", "profit", "loss", "fiscal",
            "earnings", "invoice", "budget", "cash flow", "audit",
            "assets", "liabilities", "equity", "income statement",
            "quarterly", "annual report", "ebitda", "gross margin",
            "net income", "operating expenses", "depreciation"
        ] if k in excerpt),
    }

    print(f"[classifier] Rule-based scores: {scores}")

    best = max(scores, key=scores.get)

    # Threshold lowered to 1 — one strong keyword match is enough
    # Legal documents often have very distinct vocabulary that doesn't repeat
    if scores[best] >= 1:
        print(f"[classifier] Rule-based: '{best}' (score: {scores[best]})")
        return best

    return "general"