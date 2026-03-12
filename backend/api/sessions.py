from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession
from models.database import get_db
from memory.session_store import create_session, get_session, list_sessions
from memory.chat_history import get_all_messages

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("/")
def new_session(db: DBSession = Depends(get_db)):
    session = create_session(db)
    return {"session_id": str(session.id), "created_at": session.created_at}

@router.get("/")
def all_sessions(db: DBSession = Depends(get_db)):
    sessions = list_sessions(db)
    return [
        {"session_id": str(s.id), "title": s.title, "updated_at": s.updated_at}
        for s in sessions
    ]

@router.get("/{session_id}")
def fetch_session(session_id: str, db: DBSession = Depends(get_db)):
    session = get_session(db, session_id)
    if not session:
        return {"error": "Session not found"}
    return {"session_id": str(session.id), "title": session.title}

@router.get("/{session_id}/messages")
def fetch_messages(session_id: str, db: DBSession = Depends(get_db)):
    messages = get_all_messages(db, session_id)
    return [
        {"role": m.role, "content": m.content, "created_at": m.created_at}
        for m in messages
    ]

@router.get("/{session_id}/documents")
def fetch_documents(session_id: str, db: DBSession = Depends(get_db)):
    from models.schema import Document
    docs = db.query(Document).filter(
        Document.session_id == session_id,
        Document.status.in_(["complete", "completed"])
    ).order_by(Document.created_at.asc()).all()
    return [
        {
            "filename": d.filename,
            "doc_type": d.doc_type or "general",
            "pdf_id": str(d.id),
            "status": d.status
        }
        for d in docs
    ]

@router.get("/{session_id}/documents/{pdf_id}")
def fetch_document_by_pdf_id(session_id: str, pdf_id: str, db: DBSession = Depends(get_db)):
    from models.schema import Document
    doc = db.query(Document).filter(
        Document.session_id == session_id
    ).all()
    # pdf_id in citations is the vector store pdf_id, match by position
    return [{"filename": d.filename, "doc_type": d.doc_type, "id": str(d.id)} for d in doc]