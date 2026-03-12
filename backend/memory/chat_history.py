from sqlalchemy.orm import Session as DBSession
from models.schema import Message

def save_message(db: DBSession, session_id: str, role: str, content: str, citations=None, ragas_score=None):
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        citations=citations,
        ragas_score=ragas_score
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg

def get_recent_messages(db: DBSession, session_id: str, limit: int = 10):
    msgs = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return msgs[::-1]

def get_all_messages(db: DBSession, session_id: str):
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )