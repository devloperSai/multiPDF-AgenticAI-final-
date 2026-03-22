import os
import uuid
import sys
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session as DBSession
from celery.result import AsyncResult
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

from celery_config import celery_app
from workers.pdf_worker import process_pdf
from models.database import init_db, get_db
from api.sessions import router as sessions_router
from api.qa import router as qa_router
from api.auth import router as auth_router, get_current_user  # ← auth
from models.schema import User
from core.semantic_cache import clear_cache
from api.pdf_validator import validate_pdf_bytes  # Refine 7

app = FastAPI(title="Multi-PDF QA System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# ── Warm up local models at startup ──────────────────────────────────────────
# Pre-loads NLI classifier onto GPU so first user request isn't slow.
# Non-blocking — if model fails to load, keyword fallback activates silently.
@app.on_event("startup")
async def startup_event():
    log_config()
    try:
        from graph.router import warm_up
        warm_up()
    except Exception as e:
        print(f"[startup] Router warm-up failed: {e} — keyword fallback active")

# ── routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router)       # /auth/signup  /auth/login  /auth/me
app.include_router(sessions_router)   # /sessions/...
app.include_router(qa_router)         # /qa/ask/stream

from config import cfg, log_config

UPLOAD_DIR   = cfg.UPLOAD_DIR
PDF_CAP      = cfg.PDF_CAP
PDF_EVICT_TO = cfg.PDF_EVICT_TO

os.makedirs(UPLOAD_DIR, exist_ok=True)


def _evict_oldest_pdfs(db: DBSession, exclude_session_id: str):
    """
    Global PDF eviction — called before every upload.
    If total completed PDFs >= PDF_CAP, delete ChromaDB chunks for oldest PDFs
    (excluding current session) until count drops to PDF_EVICT_TO.
    PostgreSQL records are kept with status='evicted'.

    Design:
    - Never evicts from the current active session
    - Evicts oldest by created_at across all other sessions
    - ChromaDB chunks deleted — vector search no longer works for evicted docs
    - PostgreSQL record kept so user sees eviction history
    """
    from memory.session_store import (
        count_active_pdfs,
        get_oldest_pdfs_for_eviction,
        mark_document_evicted
    )
    from core.vector_store import delete_pdf_chunks

    total = count_active_pdfs(db)

    if total < PDF_CAP:
        return

    slots_needed = total - PDF_EVICT_TO
    print(f"[eviction] Cap hit: {total} PDFs >= {PDF_CAP} — evicting {slots_needed} oldest")

    candidates = get_oldest_pdfs_for_eviction(db, exclude_session_id, limit=slots_needed)

    if not candidates:
        print(f"[eviction] No eviction candidates outside current session — skipping")
        return

    for doc in candidates:
        chroma_pdf_id = doc.chroma_pdf_id
        session_id    = str(doc.session_id)

        if chroma_pdf_id:
            try:
                deleted = delete_pdf_chunks(session_id, chroma_pdf_id)
                print(f"[eviction] Deleted {deleted} ChromaDB chunks for '{doc.filename}' (session {session_id[:8]}...)")
            except Exception as e:
                print(f"[eviction] ChromaDB cleanup failed for '{doc.filename}': {e}")
        else:
            print(f"[eviction] No chroma_pdf_id for '{doc.filename}' — ChromaDB cleanup skipped")

        mark_document_evicted(db, str(doc.id))
        clear_cache(session_id)

    print(f"[eviction] Done — evicted {len(candidates)} PDFs")


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── upload ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(
    file:         UploadFile        = File(...),
    session_id:   Optional[str]     = Form(None),
    db:           DBSession          = Depends(get_db),
    current_user: User               = Depends(get_current_user),   # ← requires JWT
):
    """
    Upload a PDF → validate → duplicate check → evict if over cap → save → dispatch.
    Refine 7  — PDF validated before saving to disk
    Refine 13 — PDF deleted from disk by worker after processing
    Refine 14 — duplicate detection via SHA256 content hash
    Eviction  — oldest PDFs evicted from ChromaDB when global cap hit
    Auth      — requires valid JWT; session ownership verified
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Refine 7 — validate PDF before saving to disk
    validate_pdf_bytes(contents, file.filename)

    # Refine 14 — compute SHA256 hash of file contents
    content_hash = hashlib.sha256(contents).hexdigest()
    print(f"[upload] Content hash: {content_hash[:16]}... for '{file.filename}'")

    doc_id = None
    if session_id:
        from memory.session_store import attach_document, get_session, find_duplicate_in_session
        session = get_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")

        # Verify session belongs to the authenticated user
        if session.user_id and str(session.user_id) != str(current_user.id):
            raise HTTPException(status_code=403, detail="Access denied to this session.")

        # Refine 14 — duplicate check by content hash
        existing = find_duplicate_in_session(db, session_id, content_hash)
        if existing:
            print(f"[upload] Duplicate detected: '{file.filename}' matches '{existing.filename}'")
            raise HTTPException(
                status_code=409,
                detail=f"This document has already been uploaded to this session as '{existing.filename}'. Duplicate files are not allowed."
            )

    # Global PDF cap eviction — runs before saving new file
    _evict_oldest_pdfs(db, exclude_session_id=session_id or "")

    # All checks passed — save to disk
    pdf_id    = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{pdf_id}.pdf")

    with open(file_path, "wb") as f:
        f.write(contents)

    if session_id:
        from memory.session_store import attach_document
        doc    = attach_document(db, session_id, file.filename, file_path, content_hash=content_hash)
        doc_id = str(doc.id)

        clear_cache(session_id)
        print(f"[upload] Semantic cache cleared for session {session_id}")

    task = process_pdf.delay(file_path, pdf_id, session_id, doc_id)

    return {
        "job_id":     task.id,
        "pdf_id":     pdf_id,
        "doc_id":     doc_id,
        "filename":   file.filename,
        "session_id": session_id,
        "message":    "PDF uploaded. Processing started.",
    }


# ── status ────────────────────────────────────────────────────────────────────

@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll this endpoint to check processing status."""
    try:
        result = AsyncResult(job_id, app=celery_app)
        state  = result.state

        if state == "PENDING":
            return {"job_id": job_id, "status": "pending", "message": "Task queued."}

        elif state in ("EXTRACTING", "CLASSIFYING", "CHUNKING", "EMBEDDING", "STORING"):
            return {
                "job_id":  job_id,
                "status":  state.lower(),
                "message": result.info.get("message", ""),
            }

        elif state == "SUCCESS":
            data = result.result
            return {
                "job_id":      job_id,
                "status":      "success",
                "pdf_id":      data["pdf_id"],
                "session_id":  data.get("session_id"),
                "doc_type":    data.get("doc_type"),
                "chunk_count": data["chunk_count"],
            }

        elif state == "FAILURE":
            return {"job_id": job_id, "status": "failed", "message": str(result.info)}

        return {"job_id": job_id, "status": state.lower()}

    except Exception as e:
        return {"job_id": job_id, "status": "error", "message": str(e)}


# ── session delete ────────────────────────────────────────────────────────────

@app.delete("/sessions/{session_id}")
def delete_session_endpoint(
    session_id:   str,
    db:           DBSession = Depends(get_db),
    current_user: User      = Depends(get_current_user),
):
    """
    Refine 16 — Full session deletion.
    Cleans: PostgreSQL + ChromaDB collection + semantic cache.
    Auth      — only the session owner can delete.
    """
    from memory.session_store import delete_session, get_session
    from core.vector_store import delete_session_collection
    from core.semantic_cache import clear_cache

    session = get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if session.user_id and str(session.user_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied to this session.")

    doc_infos = delete_session(db, session_id)

    try:
        delete_session_collection(session_id)
    except Exception as e:
        print(f"[delete_session] ChromaDB cleanup warning: {e}")

    clear_cache(session_id)

    print(f"[delete_session] Session {session_id} fully deleted ({len(doc_infos)} documents)")
    return {
        "status":            "deleted",
        "session_id":        session_id,
        "documents_removed": len(doc_infos),
    }


# ── document delete ───────────────────────────────────────────────────────────

@app.delete("/sessions/{session_id}/documents/{doc_id}")
def delete_document_endpoint(
    session_id:   str,
    doc_id:       str,
    db:           DBSession = Depends(get_db),
    current_user: User      = Depends(get_current_user),
):
    """
    Refine 16 — Full single document deletion.
    Cleans: PostgreSQL + ChromaDB chunks + semantic cache.
    Auth      — only the session owner can delete.
    """
    from memory.session_store import delete_document, get_session
    from core.vector_store import delete_pdf_chunks
    from core.semantic_cache import clear_cache

    session = get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if session.user_id and str(session.user_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied to this session.")

    info = delete_document(db, doc_id)
    if not info:
        raise HTTPException(status_code=404, detail="Document not found.")

    chroma_pdf_id = info.get("chroma_pdf_id")
    deleted_chunks = 0
    if chroma_pdf_id:
        try:
            deleted_chunks = delete_pdf_chunks(session_id, chroma_pdf_id)
            print(f"[delete_document] Deleted {deleted_chunks} chunks from ChromaDB")
        except Exception as e:
            print(f"[delete_document] ChromaDB cleanup warning: {e}")

    clear_cache(session_id)

    return {
        "status":        "deleted",
        "doc_id":        doc_id,
        "filename":      info["filename"],
        "chunks_removed": deleted_chunks,
    }