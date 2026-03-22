from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from models.database import get_db
from models.schema import Session, Document, Message
from memory.session_store import create_session, get_session
from memory.chat_history import get_all_messages
from api.auth import get_current_user, User

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/")
def new_session(
    db:           DBSession = Depends(get_db),
    current_user: User      = Depends(get_current_user),
):
    session = create_session(db, user_id=str(current_user.id))
    return {"session_id": str(session.id), "created_at": session.created_at}


@router.get("/")
def all_sessions(
    db:           DBSession = Depends(get_db),
    current_user: User      = Depends(get_current_user),
):
    sessions = (
        db.query(Session)
        .filter(Session.user_id == current_user.id)
        .order_by(Session.updated_at.desc())
        .all()
    )
    return [
        {"session_id": str(s.id), "title": s.title, "updated_at": s.updated_at}
        for s in sessions
    ]


@router.get("/{session_id}")
def fetch_session(
    session_id:   str,
    db:           DBSession = Depends(get_db),
    current_user: User      = Depends(get_current_user),
):
    session = db.query(Session).filter(
        Session.id      == session_id,
        Session.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": str(session.id), "title": session.title}


@router.delete("/{session_id}")
def delete_session(
    session_id:   str,
    db:           DBSession = Depends(get_db),
    current_user: User      = Depends(get_current_user),
):
    session = db.query(Session).filter(
        Session.id      == session_id,
        Session.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Get all chroma_pdf_ids before deleting from PostgreSQL
    docs = db.query(Document).filter(Document.session_id == session_id).all()
    chroma_pdf_ids = [d.chroma_pdf_id for d in docs if d.chroma_pdf_id]

    db.query(Message).filter(Message.session_id == session_id).delete()
    db.query(Document).filter(Document.session_id == session_id).delete()
    db.delete(session)
    db.commit()

    # Clean ChromaDB — delete entire session collection
    try:
        from core.vector_store import delete_session_collection
        delete_session_collection(session_id)
        print(f"[delete_session] ChromaDB collection deleted for session {session_id}")
    except Exception as e:
        print(f"[delete_session] ChromaDB cleanup failed: {e}")

    # Clear BM25 cache
    try:
        from core.bm25_store import invalidate_bm25_index
        invalidate_bm25_index(session_id)
    except Exception as e:
        print(f"[delete_session] BM25 invalidation failed: {e}")

    # Clear semantic cache
    try:
        from core.semantic_cache import clear_cache
        clear_cache(session_id)
    except Exception as e:
        print(f"[delete_session] Cache clear failed: {e}")

    # Clear summary memory cache — no point keeping a summary of a deleted session
    try:
        from memory.context_builder import clear_summary_cache
        clear_summary_cache(session_id)
    except Exception as e:
        print(f"[delete_session] Summary cache clear failed: {e}")

    return {"deleted": session_id}


@router.get("/{session_id}/messages")
def fetch_messages(
    session_id:   str,
    db:           DBSession = Depends(get_db),
    current_user: User      = Depends(get_current_user),
):
    session = db.query(Session).filter(
        Session.id      == session_id,
        Session.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = get_all_messages(db, session_id)
    return [
        {
            "role":       m.role,
            "content":    m.content,
            "citations":  m.citations or [],
            "created_at": m.created_at,
        }
        for m in messages
    ]


@router.get("/{session_id}/documents")
def fetch_documents(
    session_id:   str,
    db:           DBSession = Depends(get_db),
    current_user: User      = Depends(get_current_user),
):
    session = db.query(Session).filter(
        Session.id      == session_id,
        Session.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    docs = (
        db.query(Document)
        .filter(
            Document.session_id == session_id,
            Document.status.in_(["complete", "completed", "success"]),
        )
        .order_by(Document.created_at.asc())
        .all()
    )
    return [
        {
            "filename": d.filename,
            "doc_type": d.doc_type or "general",
            "pdf_id":   str(d.id),
            "status":   d.status,
        }
        for d in docs
    ]


@router.delete("/{session_id}/documents/{doc_id}")
def delete_document(
    session_id:   str,
    doc_id:       str,
    db:           DBSession = Depends(get_db),
    current_user: User      = Depends(get_current_user),
):
    session = db.query(Session).filter(
        Session.id      == session_id,
        Session.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    doc = db.query(Document).filter(
        Document.id         == doc_id,
        Document.session_id == session_id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    chroma_pdf_id = doc.chroma_pdf_id

    # 1. Delete from PostgreSQL
    db.delete(doc)
    db.commit()

    # 2. Delete chunks from ChromaDB
    if chroma_pdf_id:
        try:
            from core.vector_store import delete_pdf_chunks
            deleted = delete_pdf_chunks(session_id, chroma_pdf_id)
            print(f"[delete_doc] Removed {deleted} chunks from ChromaDB for pdf_id {chroma_pdf_id}")
        except Exception as e:
            print(f"[delete_doc] ChromaDB cleanup failed: {e}")

    # 3. Invalidate BM25 index — rebuilt fresh on next query
    try:
        from core.bm25_store import invalidate_bm25_index
        invalidate_bm25_index(session_id)
        print(f"[delete_doc] BM25 index invalidated for session {session_id}")
    except Exception as e:
        print(f"[delete_doc] BM25 invalidation failed: {e}")

    # 4. Clear semantic cache — cached answers may reference deleted doc
    try:
        from core.semantic_cache import clear_cache
        clear_cache(session_id)
        print(f"[delete_doc] Semantic cache cleared for session {session_id}")
    except Exception as e:
        print(f"[delete_doc] Cache clear failed: {e}")

    return {"deleted": doc_id}