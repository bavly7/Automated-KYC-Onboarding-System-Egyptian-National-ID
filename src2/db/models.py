"""
SQLAlchemy ORM models for KYC Verification System.
Includes User authentication and KYCSession tables for Supabase PostgreSQL.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_new_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    # Relationship to KYC Sessions
    sessions = relationship("KYCSession", back_populates="user")


class KYCSession(Base):
    __tablename__ = "kyc_sessions"

    id = Column(String, primary_key=True, default=_new_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)

    # Session Status: pending | processing | done | error
    status = Column(String, nullable=False, default="pending")

    # Liveness challenge instructions generated at session start
    liveness_instructions = Column(JSON, nullable=True)

    # Terminal state outcomes
    decision = Column(String, nullable=True)  # approved | manual_review | rejected | mobile_handoff_required
    rejection_reason = Column(String, nullable=True)
    extracted_fields = Column(JSON, nullable=True)
    face_similarity = Column(Float, nullable=True)
    requires_mobile_handoff = Column(Boolean, nullable=False, default=False)

    # National ID extracted from OCR for duplicate detection
    national_id = Column(String, nullable=True, index=True)

    # Mobile Handoff fields
    email = Column(String, nullable=True)
    handoff_token = Column(String, nullable=True)
    handoff_token_expires_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Relationship to User model
    user = relationship("User", back_populates="sessions")