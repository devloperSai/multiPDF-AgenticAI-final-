from sqlalchemy.orm import Session as DBSession
from models.schema import Session, Document


def create_session(db: DBSession) -> Session:
    session = Session()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: DBSession, session_id: str) -> Session:
    return db.query(Session).filter(Session.id == session_id).first()


def list_sessions(db: DBSession):
    return db.query(Session).order_by(Session.updated_at.desc()).all()


def attach_document(db: DBSession, session_id: str, filename: str, file_path: str, content_hash: str = None) -> Document:
    # Refine 14 — store content_hash at document creation time
    doc = Document(
        session_id=session_id,
        filename=filename,
        file_path=file_path,
        content_hash=content_hash
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def update_document_status(db: DBSession, doc_id: str, status: str, doc_type: str = None, chroma_pdf_id: str = None):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        doc.status = status
        if doc_type:
            doc.doc_type = doc_type
        if chroma_pdf_id:
            doc.chroma_pdf_id = chroma_pdf_id
        db.commit()
    return doc


def get_session_doc_types(db: DBSession, session_id: str) -> list:
    """Get all doc_types for completed documents in a session."""
    docs = db.query(Document).filter(
        Document.session_id == session_id,
        Document.status.in_(["complete", "completed"])
    ).all()
    types = list(set([d.doc_type for d in docs if d.doc_type]))
    print(f"[session_store] Doc types for session {session_id}: {types}")
    return types if types else ["general"]


def update_session_title(db: DBSession, session_id: str, question: str):
    """
    Auto-generate session title from first user question.
    Takes first 6 words, title-cased.
    Only sets title if not already set.
    """
    session = db.query(Session).filter(Session.id == session_id).first()
    if session and not session.title:
        words = question.strip().split()[:6]
        title = " ".join(words).title()
        title = title.rstrip("?!.,")
        session.title = title
        db.commit()
        print(f"[session] Title set: '{title}'")


def find_duplicate_in_session(db: DBSession, session_id: str, content_hash: str) -> Document:
    """
    Refine 14 — Check if a document with the same content hash already exists in this session.
    Returns the existing document if found, else None.
    """
    return db.query(Document).filter(
        Document.session_id == session_id,
        Document.content_hash == content_hash,
        Document.status.in_(["completed", "complete"])
    ).first()


def count_active_pdfs(db: DBSession) -> int:
    """
    Global PDF eviction — count all completed (non-evicted) PDFs across all sessions.
    """
    return db.query(Document).filter(
        Document.status.in_(["completed", "complete"])
    ).count()


def get_oldest_pdfs_for_eviction(db: DBSession, exclude_session_id: str, limit: int) -> list:
    """
    Global PDF eviction — get oldest completed PDFs across all sessions.
    Excludes the current active session so new uploads aren't immediately evicted.
    Returns list of Document objects ordered oldest first.
    """
    docs = (
        db.query(Document)
        .filter(
            Document.status.in_(["completed", "complete"]),
            Document.session_id != exclude_session_id
        )
        .order_by(Document.created_at.asc())  # oldest first
        .limit(limit)
        .all()
    )
    return docs


def mark_document_evicted(db: DBSession, doc_id: str):
    """
    Global PDF eviction — mark document as evicted in PostgreSQL.
    ChromaDB chunks already deleted by caller.
    PostgreSQL record kept so user can see what was evicted.
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        doc.status = "evicted"
        db.commit()
        print(f"[session_store] Document {doc_id} ('{doc.filename}') marked as evicted")
    return doc


def delete_document(db: DBSession, doc_id: str) -> dict:
    """
    Refine 16 — Full document deletion.
    Returns dict with chroma_pdf_id, session_id, file_path for caller to clean up.
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return None

    info = {
        "doc_id": str(doc.id),
        "session_id": str(doc.session_id),
        "chroma_pdf_id": doc.chroma_pdf_id,
        "file_path": doc.file_path,
        "filename": doc.filename
    }

    db.delete(doc)
    db.commit()
    print(f"[session_store] Document {doc_id} deleted from PostgreSQL")
    return info


def delete_session(db: DBSession, session_id: str) -> list:
    """
    Refine 16 — Full session deletion.
    Returns list of document infos for caller to clean ChromaDB + disk.
    """
    from models.schema import Message

    docs = db.query(Document).filter(Document.session_id == session_id).all()
    doc_infos = [
        {
            "doc_id": str(d.id),
            "chroma_pdf_id": d.chroma_pdf_id,
            "file_path": d.file_path,
            "filename": d.filename
        }
        for d in docs
    ]

    db.query(Message).filter(Message.session_id == session_id).delete()
    db.query(Document).filter(Document.session_id == session_id).delete()
    db.query(Session).filter(Session.id == session_id).delete()
    db.commit()

    print(f"[session_store] Session {session_id} and {len(docs)} documents deleted from PostgreSQL")
    return doc_infos