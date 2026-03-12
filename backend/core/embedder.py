from sentence_transformers import SentenceTransformer
from typing import List
import torch

# Module-level singleton — loaded once per worker process, not per task
_model: SentenceTransformer = None

# Safe batch size for GTX 1650 (4GB VRAM) — avoids OOM on large PDFs
BATCH_SIZE = 32


def _get_device() -> str:
    """
    Use CUDA if available, fall back to CPU.
    Logs which device is being used so it's visible in Celery output.
    """
    if torch.cuda.is_available():
        device = "cuda"
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[embedder] Using GPU: {torch.cuda.get_device_name(0)} ({vram:.1f}GB VRAM)")
    else:
        device = "cpu"
        print("[embedder] CUDA not available — using CPU")
    return device


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = _get_device()
        print(f"[embedder] Loading BAAI/bge-base-en-v1.5 on {device}...")
        _model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
        print("[embedder] Model loaded.")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of passage texts (chunks from PDFs).
    BGE does NOT use a prefix for passages — only for queries.
    Returns list of normalized float vectors.

    GPU path:  CUDA embedding with batch_size=32 (safe for 4GB VRAM)
    Fallback:  if CUDA OOM occurs, retries on CPU automatically
               worker does not crash — just slower for that batch
    """
    if not texts:
        return []

    model = _get_model()

    try:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=len(texts) > 50
        )
        return embeddings.tolist()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # CUDA OOM — clear cache and retry on CPU
            print(f"[embedder] CUDA out of memory — clearing cache and falling back to CPU")
            torch.cuda.empty_cache()

            # Temporarily move model to CPU for this batch
            model.to("cpu")
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=BATCH_SIZE,
                show_progress_bar=len(texts) > 50
            )
            # Move back to GPU for next call
            if torch.cuda.is_available():
                model.to("cuda")
                print("[embedder] Model moved back to GPU after OOM recovery")

            return embeddings.tolist()
        raise


def embed_query(query: str) -> List[float]:
    """
    Embed a single search query.
    BGE REQUIRES this specific prefix for queries — without it retrieval quality drops significantly.
    Do NOT use this function for passage/chunk embedding.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    model = _get_model()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"

    try:
        embedding = model.encode(prefixed, normalize_embeddings=True)
        return embedding.tolist()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("[embedder] CUDA OOM on query embedding — falling back to CPU")
            torch.cuda.empty_cache()
            model.to("cpu")
            embedding = model.encode(prefixed, normalize_embeddings=True)
            if torch.cuda.is_available():
                model.to("cuda")
            return embedding.tolist()
        raise