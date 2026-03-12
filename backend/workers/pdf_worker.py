import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celery_config import celery_app
from ingestion.extractor import extract_pages_from_pdf
from ingestion.chunker import chunk_pages
from core.embedder import embed_texts
from core.vector_store import store_chunks
from core.classifier import classify_document


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
        doc_type = classify_document(full_text)

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
                # Refine 16 — store pdf_id as chroma_pdf_id so deletion works correctly
                update_document_status(db, doc_id, "completed", doc_type, chroma_pdf_id=pdf_id)
            finally:
                db.close()

        # Refine 13 — delete PDF from disk after successful processing
        # File is no longer needed — all content is in ChromaDB
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"[pdf_worker] Deleted from disk: {pdf_path}")
        except Exception as e:
            # Non-fatal — processing succeeded, disk cleanup failed
            print(f"[pdf_worker] Warning: could not delete {pdf_path}: {e}")

        return {
            "status": "success",
            "pdf_id": pdf_id,
            "session_id": _session_id,
            "doc_type": doc_type,
            "chunk_count": stored,
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

        # Refine 13 — also delete from disk on failure — no point keeping it
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"[pdf_worker] Deleted failed PDF from disk: {pdf_path}")
        except Exception:
            pass

        self.update_state(state="FAILURE", meta={"message": str(e)})
        raise