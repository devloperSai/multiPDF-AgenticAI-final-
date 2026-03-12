import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os

CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_store"
)

_client: chromadb.PersistentClient = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        print(f"[vector_store] ChromaDB initialized at: {CHROMA_DIR}")
    return _client


def _get_collection(session_id: str):
    """
    Get or create a ChromaDB collection for a session.
    One collection per session — isolates PDFs per user session.
    """
    client = _get_client()
    name = f"s-{session_id}"
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


def store_chunks(
    session_id: str,
    chunks: List[Dict],
    embeddings: List[List[float]]
) -> int:
    """
    Upsert chunks + embeddings into session's ChromaDB collection.
    Refine 2 — includes doc_type in metadata.
    Returns count of stored chunks.
    """
    if not chunks or not embeddings:
        raise ValueError("chunks and embeddings cannot be empty.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
        )

    collection = _get_collection(session_id)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]

    metadatas = []
    for c in chunks:
        chunk_meta = c.get("metadata", {}) or {}
        meta = {
            "pdf_id": c["pdf_id"],
            "chunk_index": str(c["chunk_index"]),
            "page_number": str(chunk_meta.get("page_number", "")),
            "doc_type": str(chunk_meta.get("doc_type", "general")),
        }
        metadatas.append(meta)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return len(ids)


def query_chunks(
    session_id: str,
    query_embedding: List[float],
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieve top-k most similar chunks for a given query embedding.
    Returns list of dicts: text, metadata, similarity score.
    """
    collection = _get_collection(session_id)

    count = collection.count()
    if count == 0:
        return []

    n_results = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        similarity = round(1 - distance, 4)
        chunks.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "score": similarity
        })

    return chunks


def query_chunks_by_pdf(
    session_id: str,
    query_embedding: List[float],
    pdf_id: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieve top-k chunks filtered to a specific PDF within a session.
    Used for document comparison — fetches relevant chunks per document separately.
    """
    collection = _get_collection(session_id)

    count = collection.count()
    if count == 0:
        return []

    n_results = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"pdf_id": pdf_id},
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        similarity = round(1 - distance, 4)
        chunks.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "score": similarity
        })

    return chunks


def get_session_pdf_ids(session_id: str) -> List[str]:
    """
    Get all unique pdf_ids stored in a session's collection.
    Used by comparison retrieval to know which PDFs are available.
    """
    collection = _get_collection(session_id)

    count = collection.count()
    if count == 0:
        return []

    results = collection.get(include=["metadatas"])

    pdf_ids = list(set(
        m["pdf_id"] for m in results["metadatas"] if m.get("pdf_id")
    ))

    print(f"[vector_store] Session {session_id} has {len(pdf_ids)} PDFs: {pdf_ids}")
    return pdf_ids


def delete_pdf_chunks(session_id: str, pdf_id: str) -> int:
    """
    Refine 10 — Delete all chunks belonging to a specific PDF from a session's collection.
    Returns count deleted.
    """
    collection = _get_collection(session_id)

    results = collection.get(where={"pdf_id": pdf_id})
    ids_to_delete = results["ids"]

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"[vector_store] Deleted {len(ids_to_delete)} chunks for pdf_id {pdf_id}")

    return len(ids_to_delete)


def delete_session_collection(session_id: str):
    """
    Refine 16 — Delete the entire ChromaDB collection for a session.
    Called when a session is fully deleted.
    """
    client = _get_client()
    collection_name = f"s-{session_id}"

    try:
        client.delete_collection(collection_name)
        print(f"[vector_store] Deleted collection '{collection_name}'")
    except Exception as e:
        # Collection may not exist if session had no PDFs processed
        print(f"[vector_store] Collection '{collection_name}' not found or already deleted: {e}")