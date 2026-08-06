"""
SQLAlchemy ORM models. One table for now — KYCSession — since Phase 4
scope is "make duplicate_check_node real + let the frontend poll a
result", not a full multi-table schema. Extend later (e.g. a separate
audit/event log table) if you need a history rather than just current state.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class KYCSession(Base):
    __tablename__ = "kyc_sessions"

    id = Column(String, primary_key=True, default=_new_uuid)

    # "pending"   -> created via POST /sessions, waiting for upload
    # "processing"-> upload received, graph running in the background
    # "done"      -> graph finished, `decision` below is authoritative
    # "error"     -> graph raised an exception (see rejection_reason)
    status = Column(String, nullable=False, default="pending")

    # Chosen at session-creation time (Phase 4/5 hybrid liveness) and
    # handed to the frontend so it knows what to display/capture. Stored
    # here (not just returned once) so /status or a retry can recover them.
    liveness_instructions = Column(JSON, nullable=True)

    # Populated once the graph reaches a terminal state.
    decision = Column(String, nullable=True)  # approved | manual_review | rejected | mobile_handoff_required
    rejection_reason = Column(String, nullable=True)
    extracted_fields = Column(JSON, nullable=True)
    face_similarity = Column(Float, nullable=True)
    requires_mobile_handoff = Column(Boolean, nullable=False, default=False)

    # Indexed implicitly via the duplicate_check_node query pattern; add an
    # explicit index if this table grows large enough for it to matter.
    national_id = Column(String, nullable=True)

    # ---- Mobile handoff (email + continue-on-another-device) ----
    # email is only collected once the user actually needs a handoff — not
    # asked for up front. handoff_token is single-use: cleared the moment
    # /sessions/{id}/resume consumes it, so a leaked/reused link can't
    # replay an old session.
    email = Column(String, nullable=True)
    handoff_token = Column(String, nullable=True)
    handoff_token_expires_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
