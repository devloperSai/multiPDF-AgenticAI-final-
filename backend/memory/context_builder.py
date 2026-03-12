from sqlalchemy.orm import Session as DBSession
from memory.chat_history import get_recent_messages

def build_memory_context(db: DBSession, session_id: str, limit: int = 8) -> str:
    messages = get_recent_messages(db, session_id, limit)
    if not messages:
        return ""
    lines = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)