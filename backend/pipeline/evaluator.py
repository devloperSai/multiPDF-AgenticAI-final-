import math
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from typing import List, Optional, Dict
import os
from dotenv import load_dotenv

load_dotenv()

_llm = None
_embeddings = None


def _get_llm():
    global _llm
    if _llm is None:
        groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_tokens=4096,
        )
        _llm = LangchainLLMWrapper(groq_llm)
    return _llm


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )
        _embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    return _embeddings


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


def evaluate_response(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, Optional[float]]:
    
    try:
        if not answer or not contexts:
            return {"faithfulness": None, "answer_relevancy": None}

        contexts = [c for c in contexts if c and c.strip()]
        if not contexts:
            return {"faithfulness": None, "answer_relevancy": None}

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })

        llm = _get_llm()
        embeddings = _get_embeddings()

        # Only use faithfulness — answer_relevancy requires n>1 (not supported by Groq)
        faithfulness.llm = llm
        faithfulness.embeddings = embeddings

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
        )

        result_df = result.to_pandas()
        faith_val = _safe_float(result_df["faithfulness"].iloc[0])

        scores = {
            "faithfulness": faith_val,
            "answer_relevancy": None  # skipped — Groq doesn't support n>1
        }

        print(f"[evaluator] faithfulness={scores['faithfulness']}")
        return scores

    except Exception as e:
        print(f"[evaluator] RAGAS evaluation failed: {e}")
        return {"faithfulness": None, "answer_relevancy": None}