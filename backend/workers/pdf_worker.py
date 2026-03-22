import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celery_config import celery_app
from ingestion.extractor import extract_pages_from_pdf
from ingestion.chunker import chunk_pages
from core.embedder import embed_texts
from core.vector_store import store_chunks
from core.classifier import classify_document


def _generate_doc_summary(full_text: str, doc_type: str, filename: str) -> str | None:
    """
    Enhancement #15 — Generate a 2-3 sentence document summary at ingestion time.

    WHY at ingestion:
        When user asks "what is this document about?" or "give me a brief overview",
        we already know the answer — we read the whole document during processing.
        Storing it once means zero retrieval needed for these vague questions.

    WHAT the summary covers:
        - What type of document it is
        - Main topic or subject matter
        - Key parties, concepts, or scope

    Example output:
        "This is a professional services contract between Santa Cruz Regional
         Transportation Commission and a consultant for shuttle services.
         It covers payment terms, insurance requirements, termination clauses,
         and record-keeping obligations over a defined contract period."

    Uses call_with_fallback — benefits from full provider chain.
    Falls back gracefully — non-fatal if all providers fail.
    Uses first 3000 chars of text — enough context, avoids token waste.
    """
    try:
        from graph.fallback_llm import call_with_fallback

        # Use first 3000 chars — covers title, parties, main clauses
        excerpt = full_text[:3000].strip()
        if len(excerpt) < 100:
            return None

        doc_type_hint = {
            "legal":     "Focus on: parties involved, subject matter, key obligations and rights.",
            "research":  "Focus on: research topic, methodology, main findings and conclusions.",
            "financial": "Focus on: financial subject, key figures, time period and entities involved.",
            "general":   "Focus on: main topic, key information, and purpose of the document.",
        }.get(doc_type, "Focus on: main topic and purpose.")

        summary, provider = call_with_fallback(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document summarizer. "
                        "Write a 2-3 sentence summary of the document. "
                        f"{doc_type_hint} "
                        "Be concise and factual. "
                        "Do not use phrases like 'this document' — just state the facts directly. "
                        "Return only the summary, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": f"Document filename: {filename}\n\nDocument text (first section):\n{excerpt}\n\nSummary:"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=150,
        )

        summary = summary.strip()
        print(f"[pdf_worker] Summary generated via {provider} ({len(summary)} chars): {summary[:80]}...")
        return summary

    except Exception as e:
        print(f"[pdf_worker] Summary generation failed (non-fatal): {e}")
        return None


@celery_app.task(bind=True, name="workers.pdf_worker.process_pdf")
def process_pdf(self, pdf_path: str, pdf_id: str, session_id: str = None, doc_id: str = None):
    """
    Celery task: Extract → Classify → Chunk → Embed → Store → Delete from disk.
    Single PDF visit — pages extracted once, reused for classification and chunking.
    Refine 13 — PDF deleted from disk after successful processing.
    """
    try:
        # Step 1: Extract pages ONCE
        self.update_state(
            state="EXTRACTING",
            meta={"message": "Extracting text from PDF..."}
        )
        pages = extract_pages_from_pdf(pdf_path)

        if not pages:
            raise ValueError(
                "No pages extracted. "
                "PDF may be scanned or image-based."
            )

        full_text = "\n\n".join(p["text"] for p in pages)

        if len(full_text.strip()) < 50:
            raise ValueError(
                "Extracted text too short. "
                "PDF may be scanned or image-based."
            )

        # Step 2: Classify
        self.update_state(
            state="CLASSIFYING",
            meta={"message": "Classifying document type..."}
        )
        doc_type = classify_document(full_text, filename=os.path.basename(pdf_path))

        if doc_id:
            from models.database import SessionLocal
            from memory.session_store import update_document_status
            db = SessionLocal()
            try:
                update_document_status(db, doc_id, "processing", doc_type)
            finally:
                db.close()

        # Step 3: Chunk
        self.update_state(
            state="CHUNKING",
            meta={"message": f"Splitting [{doc_type}] document into chunks..."}
        )
        chunks = chunk_pages(pages, pdf_id)

        if not chunks:
            raise ValueError("No chunks produced. Check PDF content.")

        # Refine 2 — inject doc_type into every chunk metadata
        for chunk in chunks:
            chunk["metadata"]["doc_type"] = doc_type

        # Step 4: Embed
        self.update_state(
            state="EMBEDDING",
            meta={"message": f"Embedding {len(chunks)} chunks..."}
        )
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        # Step 5: Store
        self.update_state(
            state="STORING",
            meta={"message": "Storing in vector database..."}
        )
        _session_id = session_id or pdf_id
        stored = store_chunks(_session_id, chunks, embeddings)

        if doc_id:
            from models.database import SessionLocal
            from memory.session_store import update_document_status
            db = SessionLocal()
            try:
                update_document_status(db, doc_id, "completed", doc_type, chroma_pdf_id=pdf_id)
            finally:
                db.close()

        # Step 6: Generate document summary
        # 2-3 sentence summary stored in PostgreSQL.
        # Used by generate_node for vague questions like "what is this document about?"
        # Non-fatal — if summary generation fails, processing still succeeds.
        self.update_state(
            state="STORING",
            meta={"message": "Generating document summary..."}
        )
        doc_summary = _generate_doc_summary(full_text, doc_type, os.path.basename(pdf_path))
        if doc_summary and doc_id:
            from models.database import SessionLocal
            from memory.session_store import save_document_summary
            db = SessionLocal()
            try:
                save_document_summary(db, doc_id, doc_summary)
            finally:
                db.close()

        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"[pdf_worker] Deleted from disk: {pdf_path}")
        except Exception as e:
            print(f"[pdf_worker] Warning: could not delete {pdf_path}: {e}")

        return {
            "status":      "success",
            "pdf_id":      pdf_id,
            "session_id":  _session_id,
            "doc_type":    doc_type,
            "chunk_count": stored,
            "has_summary": doc_summary is not None,
        }

    except Exception as e:
        if doc_id:
            from models.database import SessionLocal
            from memory.session_store import update_document_status
            db = SessionLocal()
            try:
                update_document_status(db, doc_id, "failed")
            finally:
                db.close()

       
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"[pdf_worker] Deleted failed PDF from disk: {pdf_path}")
        except Exception:
            pass

        self.update_state(state="FAILURE", meta={"message": str(e)})
        raise