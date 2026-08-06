import os
import random
import secrets
import shutil
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from src2 import config
from src2.graph.build_graph import build_kyc_graph
from src2.graph.nodes import KYCEngines, liveness_check_node  
from src2.db.session import SessionLocal, init_db, get_db
from src2.db.models import KYCSession
from src2.backend.email_utils import send_handoff_email

UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", "uploads"))

app = FastAPI(title="KYC Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    app.state.engines = KYCEngines.load()
    app.state.graph = build_kyc_graph(app.state.engines, SessionLocal)


@app.post("/sessions")
def create_session(db: Session = Depends(get_db)):
    instructions = random.sample(config.LIVENESS_INSTRUCTIONS, config.LIVENESS_NUM_CHALLENGES)
    session_row = KYCSession(status="pending", liveness_instructions=instructions)
    db.add(session_row)
    db.commit()
    db.refresh(session_row)
    return {"session_id": session_row.id, "liveness_instructions": instructions}


def _save_upload(file: UploadFile, dest_dir: Path, keep_original_name: bool = False) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)

    if keep_original_name:
        dest_path = dest_dir / file.filename
    else:
        dest_path = dest_dir / f"{uuid.uuid4().hex}_{file.filename}"
        
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(dest_path)


# ---------------------------------------------------------------------------
# POST /sessions/{id}/liveness — Fail-Fast Liveness Check
# ---------------------------------------------------------------------------
@app.post("/sessions/{session_id}/liveness")
def verify_liveness(
    session_id: str,
    liveness_frames: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    row = db.query(KYCSession).filter(KYCSession.id == session_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if len(liveness_frames) != len(row.liveness_instructions or []):
        raise HTTPException(status_code=400, detail="Invalid number of liveness frames sent.")

    session_dir = UPLOAD_ROOT / session_id
    liveness_dir = session_dir / "liveness"
    
    if liveness_dir.exists():
        shutil.rmtree(liveness_dir)
        
    liveness_paths = [_save_upload(f, liveness_dir, keep_original_name=True) for f in liveness_frames]
    
    liveness_func = liveness_check_node(app.state.engines)
    temp_state = {
        "liveness_frame_paths": liveness_paths,
        "liveness_instructions": row.liveness_instructions
    }
    
    try:
        result = liveness_func(temp_state)
        if result.get("decision") == "needs_retake":
            shutil.rmtree(liveness_dir)
            raise HTTPException(status_code=400, detail="Liveness verification failed. Make sure your movements match the instructions.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine error: {str(e)}")
        
    return {"status": "success", "message": "Liveness verified"}


# ---------------------------------------------------------------------------
# POST /sessions/{id}/upload — Complete Documents
# ---------------------------------------------------------------------------
def _run_pipeline(session_id: str, initial_state: dict):
    db = SessionLocal()
    try:
        graph = app.state.graph
        final_state = graph.invoke(initial_state)

        row = db.query(KYCSession).filter(KYCSession.id == session_id).first()
        if row is None:
            return 

        row.status = "done"
        row.decision = final_state.get("decision")
        row.rejection_reason = final_state.get("rejection_reason")
        row.extracted_fields = final_state.get("extracted_fields")
        row.face_similarity = final_state.get("face_similarity")
        row.requires_mobile_handoff = bool(final_state.get("requires_mobile_handoff", False))
        row.national_id = (final_state.get("extracted_fields") or {}).get("ID")
        db.commit()
    except Exception:
        row = db.query(KYCSession).filter(KYCSession.id == session_id).first()
        if row is not None:
            row.status = "error"
            row.rejection_reason = traceback.format_exc(limit=3)
            db.commit()
    finally:
        db.close()


@app.post("/sessions/{session_id}/upload")
def upload_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    front_frames: List[UploadFile] = File(...),
    back_frames: List[UploadFile] = File(...),
    selfie: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    row = db.query(KYCSession).filter(KYCSession.id == session_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if row.status != "pending":
        raise HTTPException(status_code=409, detail=f"Session already {row.status}")

    session_dir = UPLOAD_ROOT / session_id
    liveness_dir = session_dir / "liveness"
    
    if not liveness_dir.exists():
        raise HTTPException(status_code=400, detail="Liveness frames missing. Please complete the liveness step first.")

    liveness_paths = sorted([str(p) for p in liveness_dir.glob("*")])

    front_paths = [_save_upload(f, session_dir / "front") for f in front_frames]
    back_paths = [_save_upload(f, session_dir / "back") for f in back_frames]
    selfie_path = _save_upload(selfie, session_dir / "selfie")

    initial_state = {
        "id_front_frame_paths": front_paths,
        "id_back_frame_paths": back_paths,
        "selfie_frame_path": selfie_path,
        "liveness_frame_paths": liveness_paths,
        "liveness_instructions": row.liveness_instructions,
        "confirmation_retry_count": 0,
    }

    row.status = "processing"
    db.commit()

    background_tasks.add_task(_run_pipeline, session_id, initial_state)
    return {"session_id": session_id, "status": "processing"}


class HandoffRequest(BaseModel):
    email: EmailStr
    target: Literal["this_device", "another_device"]


def _reset_session_for_retry(row: KYCSession, db: Session) -> list:
    new_instructions = random.sample(config.LIVENESS_INSTRUCTIONS, config.LIVENESS_NUM_CHALLENGES)
    row.status = "pending"
    row.decision = None
    row.rejection_reason = None
    row.extracted_fields = None
    row.face_similarity = None
    row.requires_mobile_handoff = False
    row.liveness_instructions = new_instructions
    db.commit()
    return new_instructions


@app.post("/sessions/{session_id}/handoff")
def request_handoff(session_id: str, body: HandoffRequest, db: Session = Depends(get_db)):
    row = db.query(KYCSession).filter(KYCSession.id == session_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    row.email = body.email

    if body.target == "this_device":
        instructions = _reset_session_for_retry(row, db)
        return {"session_id": session_id, "liveness_instructions": instructions}

    token = secrets.token_urlsafe(24)
    row.handoff_token = token
    row.handoff_token_expires_at = datetime.now(timezone.utc) + timedelta(minutes=config.HANDOFF_TOKEN_TTL_MINUTES)
    _reset_session_for_retry(row, db)

    base_url = getattr(config, "FRONTEND_BASE_URL", "http://192.168.100.13:5500")
    link = f"{base_url}?session_id={session_id}&token={token}"
    
    send_handoff_email(body.email, link)

    return {"status": "sent", "email": body.email}


@app.get("/sessions/{session_id}/resume")
def resume_session(session_id: str, token: str, db: Session = Depends(get_db)):
    row = db.query(KYCSession).filter(KYCSession.id == session_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not row.handoff_token or row.handoff_token != token:
        raise HTTPException(status_code=403, detail="Invalid or already-used link")
    if row.handoff_token_expires_at and row.handoff_token_expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=403, detail="This link has expired")

    row.handoff_token = None
    row.handoff_token_expires_at = None
    db.commit()

    return {"session_id": row.id, "liveness_instructions": row.liveness_instructions}


@app.get("/sessions/{session_id}/status")
def get_status(session_id: str, db: Session = Depends(get_db)):
    row = db.query(KYCSession).filter(KYCSession.id == session_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": row.id,
        "status": row.status,
        "decision": row.decision,
        "rejection_reason": row.rejection_reason,
        "extracted_fields": row.extracted_fields,
        "face_similarity": row.face_similarity,
        "requires_mobile_handoff": row.requires_mobile_handoff,
    }