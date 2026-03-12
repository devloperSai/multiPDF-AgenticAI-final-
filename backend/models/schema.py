from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
import uuid, datetime

Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    doc_type = Column(String(100), nullable=True)
    file_path = Column(String(500), nullable=False)
    status = Column(String(50), default="pending")
    meta_data = Column(JSON, nullable=True)
    # Refine 14 — SHA256 hash of file contents for duplicate detection
    # Nullable so existing rows without hash are not broken
    content_hash = Column(String(64), nullable=True)
    # Refine 16 — store the ChromaDB pdf_id separately from PostgreSQL doc.id
    # These are two different UUIDs — chroma_pdf_id is what was stored in vector store
    chroma_pdf_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    ragas_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)