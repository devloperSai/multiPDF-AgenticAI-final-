"""
citations_builder.py  — backend/core/citations_builder.py

Centralises citation building so nodes.py and qa.py
both produce identical citation objects with real filenames.
"""

from typing import List, Dict
from core.vector_store import _clean_excerpt


def get_pdf_filenames_bulk(pdf_ids: List[str]) -> Dict[str, str]:
    """
    Bulk lookup: chroma_pdf_id -> filename from PostgreSQL Documents table.
    ChromaDB stores pdf_id which maps to Document.chroma_pdf_id — NOT Document.id.
    Returns dict: {chroma_pdf_id: filename}
    """
    if not pdf_ids:
        return {}
    try:
        from models.database import SessionLocal
        from models.schema import Document

        db = SessionLocal()
        try:
            docs = db.query(Document).filter(
                Document.chroma_pdf_id.in_(pdf_ids)   # ← correct column
            ).all()
            result = {doc.chroma_pdf_id: doc.filename for doc in docs if doc.filename}
            print(f"[citations] Resolved {len(result)}/{len(pdf_ids)} filenames from DB")
            return result
        finally:
            db.close()
    except Exception as e:
        print(f"[citations] get_pdf_filenames_bulk failed: {e}")
        return {}


def build_citations(chunks: List[Dict]) -> List[Dict]:
    """
    Build citation objects from retrieved + reranked chunks.

    Resolves real filenames from PostgreSQL via bulk lookup (1 DB query
    for all pdf_ids instead of N queries).

    Returns list of citation dicts ready to be:
    - saved to PostgreSQL messages.citations (JSON)
    - sent to frontend in SSE 'done' event
    - displayed in CitationModal
    """
    if not chunks:
        return []

    # Collect all unique pdf_ids — one DB round trip
    pdf_ids = list({c["metadata"].get("pdf_id") for c in chunks if c["metadata"].get("pdf_id")})
    filename_map = get_pdf_filenames_bulk(pdf_ids)
    print(f"[citations_debug] pdf_ids from chunks: {pdf_ids}")
    print(f"[citations_debug] filename_map result: {filename_map}")

    citations = []
    for i, c in enumerate(chunks):
        meta = c.get("metadata", {})
        pdf_id = meta.get("pdf_id", "")

        # Resolve real filename; fall back gracefully
        filename = filename_map.get(str(pdf_id)) or filename_map.get(pdf_id)
        if not filename:
            filename = f"Document ({str(pdf_id)[:8]}...)" if pdf_id else "Unknown"

        # page_number stored as string in ChromaDB — convert back to int if possible
        raw_page = meta.get("page_number", "")
        try:
            page = int(raw_page) if raw_page not in ("", "None", None) else None
        except (ValueError, TypeError):
            page = None

        citations.append({
            "source_index": i + 1,
            "pdf_id": pdf_id,
            "filename": filename,
            "page": page,                          # int or None — frontend uses citation.page
            "score": round(c.get("score", 0), 4),  # 0-1 cosine similarity
            "excerpt": _clean_excerpt(c.get("text", ""), max_len=300),
            "doc_type": meta.get("doc_type", "general"),
        })

    return citations