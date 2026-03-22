from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
import uuid, datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name          = Column(String(255), nullable=False)
    email         = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"
    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id    = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    title      = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"
    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id    = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    filename      = Column(String(255), nullable=False)
    doc_type      = Column(String(100), nullable=True)
    file_path     = Column(String(500), nullable=False)
    status        = Column(String(50), default="pending")
    meta_data     = Column(JSON, nullable=True)
    content_hash  = Column(String(64),  nullable=True)   # SHA256 for duplicate detection
    chroma_pdf_id = Column(String(255), nullable=True)   # ChromaDB reference for chunk deletion
    doc_summary   = Column(Text, nullable=True)           # Enhancement #15 — auto-generated summary at ingestion
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id  = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    role        = Column(String(20), nullable=False)
    content     = Column(Text, nullable=False)
    citations   = Column(JSON, nullable=True)
    ragas_score = Column(Float, nullable=True)
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)