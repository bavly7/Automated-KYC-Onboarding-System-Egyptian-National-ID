"""
LangGraph node functions. Each takes a KYCState dict and returns a dict of
updates (LangGraph merges this into state).

Engines (YOLO/OCR/Face/Liveness models) are expensive to load, so they're
constructed once and injected via KYCEngines, not re-created per node call.
"""
import re
from dataclasses import dataclass
from datetime import datetime

import cv2

from src2 import config
from src2.models import CardDetector, FieldDetector, OCREngine, FaceEngine, LivenessChecker, ConsolidationAgent
from src2.utils import blur_score, select_best_frames, consensus_per_field
from src2.graph.state import KYCState
from src2.db.models import KYCSession  # local import: keep graph/ importable without DB deps at module load


@dataclass
class KYCEngines:
    card_detector: CardDetector
    field_detector: FieldDetector
    ocr_engine: OCREngine
    face_engine: FaceEngine
    liveness_checker: LivenessChecker
    consolidation_agent: ConsolidationAgent

    @classmethod
    def load(cls):
        """Load all models once. Call this at app/process startup, not per-request."""
        return cls(
            card_detector=CardDetector(),
            field_detector=FieldDetector(),
            ocr_engine=OCREngine(),
            face_engine=FaceEngine(),
            liveness_checker=LivenessChecker(),
            consolidation_agent=ConsolidationAgent(),
        )


# ---------------------------------------------------------------------------
# for phase 4 for check liveness
# ---------------------------------------------------------------------------
def liveness_check_node(engines: KYCEngines):
    def _node(state: KYCState) -> dict:
        frame_paths = state.get("liveness_frame_paths") or []
        instructions = state.get("liveness_instructions") or []

        if not frame_paths or len(frame_paths) != len(instructions):
            return {
                "liveness_passed": False,
                "liveness_pass_count": 0,
                "decision": "needs_retake",
                "rejection_reason": "Liveness frames/instructions missing or mismatched",
            }

        pass_count = 0
        for path, instruction in zip(frame_paths, instructions):
            frame_bgr = cv2.imread(path)
            if frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if engines.liveness_checker.check_instruction(frame_rgb, instruction):
                pass_count += 1

        passed = pass_count >= config.LIVENESS_PASS_THRESHOLD
        result = {"liveness_passed": passed, "liveness_pass_count": pass_count}
        if not passed:
            result["decision"] = "needs_retake"
            result["rejection_reason"] = f"Liveness check failed ({pass_count}/{len(instructions)} challenges passed)"
        return result
    return _node



# ---------------------------------------------------------------------------
# 1-2. ID capture nodes (front / back) — multi-frame quality gate
# ---------------------------------------------------------------------------

def _process_id_side(engines: KYCEngines, frame_paths: list) -> dict:
    """
    Shared logic for front/back: crop card from each frame, detect fields
    on the sharpest frame (field boxes are stable across frames of the
    same side, so we don't need to re-detect on all 10), then run
    enhanced-only OCR per frame and return per-field consensus.

    NOTE: raw (unenhanced) OCR was tested head-to-head against enhanced
    via the majority-vote experiment (see project notes) and enhanced won
    across the tested fields — so the pipeline now runs enhanced OCR only.
    `OCREngine.extract_raw` still exists if you want to re-run that
    comparison later (e.g. once you have more labeled cards).

    "too_blurry" in the return value is what actually answers "how blurry
    is too blurry?": select_best_frames only ranks the batch of captured
    frames against EACH OTHER, so it always returns *a* sharpest frame
    even if all 10 are bad. We additionally check that sharpest frame
    against config.BLUR_THRESHOLD (an absolute floor) before trusting it
    — if it fails, the caller treats this the same as "no card detected"
    and routes to needs_retake.
    """
    crops = []
    for path in frame_paths:
        crop = engines.card_detector.detect_and_crop(path)
        if crop is not None:
            crops.append(crop)

    if not crops:
        return {"card_crop": None, "field_boxes": {}, "enhanced_reads": {}, "too_blurry": False}

    best_crops = select_best_frames(crops, top_n=min(4, len(crops)))
    reference_crop = best_crops[0]

    if blur_score(reference_crop) < config.BLUR_THRESHOLD:
        # Even the sharpest frame in the batch isn't sharp enough to trust —
        # don't bother running field detection / OCR on a bad crop.
        return {"card_crop": None, "field_boxes": {}, "enhanced_reads": {}, "too_blurry": True}

    field_boxes = engines.field_detector.detect_fields(reference_crop)

    enhanced_reads = {f: [] for f in config.TARGET_CLASSES}

    for crop in crops:
        enhanced_extracted = engines.ocr_engine.extract_enhanced(crop, field_boxes)
        for f in config.TARGET_CLASSES:
            enhanced_reads[f].append(enhanced_extracted.get(f))

    return {
        "card_crop": reference_crop,
        "field_boxes": field_boxes,
        "enhanced_reads": enhanced_reads,
        "too_blurry": False,
    }


def capture_id_front_node(engines: KYCEngines):
    def _node(state: KYCState) -> dict:
        result = _process_id_side(engines, state["id_front_frame_paths"])
        if result["card_crop"] is None:
            reason = (
                "Front-side frames too blurry to read reliably"
                if result["too_blurry"]
                else "No card detected in front-side frames"
            )
            return {"decision": "needs_retake", "rejection_reason": reason}
        return {
            "front_card_crop": result["card_crop"],
            "front_field_boxes": result["field_boxes"],
            "enhanced_reads": result["enhanced_reads"],
        }
    return _node


def capture_id_back_node(engines: KYCEngines):
    def _node(state: KYCState) -> dict:
        result = _process_id_side(engines, state["id_back_frame_paths"])
        if result["card_crop"] is None:
            reason = (
                "Back-side frames too blurry to read reliably"
                if result["too_blurry"]
                else "No card detected in back-side frames"
            )
            return {"decision": "needs_retake", "rejection_reason": reason}
        # Merge per-field, not per-key-replace: the old {**a, **b} approach
        # let back's all-None entries for front-only fields (ID, First_Name,
        # Last_Name, Add1) silently overwrite front's real OCR results,
        # since both dicts contain every TARGET_CLASSES key regardless of
        # whether that side actually detected the field. Concatenating the
        # per-field lists instead preserves both sides' real reads.
        front_reads = state.get("enhanced_reads", {})
        back_reads = result["enhanced_reads"]
        merged_enhanced = {
            f: front_reads.get(f, []) + back_reads.get(f, [])
            for f in config.TARGET_CLASSES
        }
        return {
            "back_card_crop": result["card_crop"],
            "back_field_boxes": result["field_boxes"],
            "enhanced_reads": merged_enhanced,
        }
    return _node


# ---------------------------------------------------------------------------
# 3. OCR consensus (raw vs enhanced majority vote per field)
# ---------------------------------------------------------------------------

def ocr_consensus_node(state: KYCState) -> dict:
    """
    Majority-vote consensus across the enhanced-OCR reads from all
    captured frames. Enhanced beat raw in head-to-head testing against
    ground truth (see project notes) — raw is no longer run in the
    pipeline, so there's nothing left to choose between here.
    """
    consensus = consensus_per_field(state.get("enhanced_reads", {}))
    return {"extracted_fields": consensus}


# ---------------------------------------------------------------------------
# 4. Expiry check (early exit / mobile loop breaker / fallback to issue date)
# ---------------------------------------------------------------------------

def expiry_check_node(state: KYCState) -> dict:
    extracted_fields = state.get("extracted_fields", {})
    exp_raw = extracted_fields.get("ExpDate")
    
    # Check what key your config.TARGET_CLASSES uses for the issue date
    # Adjust "IssueDate" below if your config uses something like "Issue_Date"
    issue_raw = extracted_fields.get("IssueDate") 
    
    # We need this flag to know if the user is already on a mobile retry
    is_mobile_retry = state.get("is_mobile_retry", False)

    exp_date = None
    
    # 1. First attempt: Try to parse the Expiry Date directly
    if exp_raw:
        exp_date = _parse_arabic_id_date(exp_raw)

    # 2. OCR Fallback: If Expiry Date failed, try to parse Issue Date and add 7 years
    if not exp_date and issue_raw:
        issue_date = _parse_arabic_id_date(issue_raw)
        if issue_date:
            try:
                # Egyptian IDs are valid for exactly 7 years from issuance
                exp_date = issue_date.replace(year=issue_date.year + 7)
            except ValueError:
                # Edge case handling: If issue date was Leap Day (Feb 29), 
                # shifting 7 years lands on a non-leap year, which throws an error.
                exp_date = issue_date.replace(year=issue_date.year + 7, day=28)

    # 3. If both dates are completely unreadable
    if not exp_date:
        if is_mobile_retry:
            # Loop Breaker: User already tried on mobile and OCR still failed.
            # Send to a human agent instead of causing an infinite loop.
            return {
                "expiry_valid": None,
                "decision": "manual_review",
                "rejection_reason": "Could not read ID dates clearly even after mobile retry. Forwarded to agent."
            }
        else:
            # First attempt failed: Route to email handoff to try on a better camera
            return {
                "expiry_valid": None,
                "decision": "needs_retake",
                "rejection_reason": "Could not read ID dates. Please retry using your phone camera for better quality."
            }

    # 4. We successfully determined the expiry date. Check if it's actually valid.
    is_valid = exp_date >= datetime.now().date()
    result = {"expiry_valid": is_valid}
    
    if not is_valid:
        # Date was read successfully, but the ID is genuinely expired in the past.
        result["decision"] = "rejected"
        result["rejection_reason"] = f"ID expired on {exp_date.strftime('%Y-%m-%d')}."
        
    return result


def _parse_arabic_id_date(raw: str):
    """
    Egyptian ID expiry dates have shown up in two different OCR shapes so far:
      - separated digit groups, e.g. "24/01/2028" -> re.findall gives
        3 separate matches: ["24", "01", "2028"] (DD/MM/YYYY)
      - one continuous block with no separators at all, e.g.
        "البطاقة سارية حتى ٢٠٢٨٠١٢٤" -> "...20280124" -> ONE match (YYYYMMDD)
      - NEW: Issue dates often only have Month and Year, e.g., "01/2021".
    """
    arabic_to_latin = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    normalized = raw.translate(arabic_to_latin)
    digits = re.findall(r"\d+", normalized)

    if len(digits) >= 3:
        # Separated groups: DD/MM/YYYY
        day, month, year = (int(d) for d in digits[:3])
    elif len(digits) == 2:
        # Issue dates usually just have Month and Year (2 matches)
        if len(digits[0]) == 4:
            year, month = int(digits[0]), int(digits[1])
        else:
            month, year = int(digits[0]), int(digits[1])
        day = 1 # Default to the 1st of the month if day is missing
    elif len(digits) == 1 and len(digits[0]) == 8:
        # One continuous 8-digit block: YYYYMMDD
        block = digits[0]
        year, month, day = int(block[0:4]), int(block[4:6]), int(block[6:8])
    else:
        return None

    # Basic bounds checking to prevent datetime crashes
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        return None

    try:
        return datetime(year, month, day).date()
    except ValueError:
        return None

# ---------------------------------------------------------------------------
# 5. Duplicate check — STUB, needs Phase 4 DB connection
# ---------------------------------------------------------------------------
def duplicate_check_node(session_factory):
    """
    session_factory: a zero-arg callable returning a DB session/context
    manager (i.e. `db.session.SessionLocal`), injected the same way engines
    are — so this stays testable without a live DB by passing a fake
    factory, matching how KYCEngines is injected everywhere else.

    Only checks national ID against previously APPROVED sessions. A close
    face-embedding match against other applicants is a real fraud-detection
    feature but a bigger scope item (needs a vector index / pgvector,
    similarity threshold calibration) — left as a TODO rather than guessed at.
    """
    

    def _node(state: KYCState) -> dict:
        national_id = state.get("extracted_fields", {}).get("ID")
        if not national_id:
            # Shouldn't happen — expiry_check_node already short-circuits
            # incomplete extractions to needs_retake before this node runs.
            return {"is_duplicate": False}

        with session_factory() as db:
            existing = (
                db.query(KYCSession)
                .filter(KYCSession.national_id == national_id, KYCSession.decision == "approved")
                .first()
            )
        return {"is_duplicate": existing is not None}

    return _node


# ---------------------------------------------------------------------------
# 6-7. Liveness + selfie capture, face embeddings, face match
# ---------------------------------------------------------------------------

def face_match_node(engines: KYCEngines):
    def _node(state: KYCState) -> dict:
        id_face_crop = state["front_card_crop"]  # InsightFace finds the face within this
        selfie_img = cv2.imread(state["selfie_frame_path"])

        try:
            id_embedding = engines.face_engine.get_embedding(id_face_crop, label="ID photo")
            selfie_embedding = engines.face_engine.get_embedding(selfie_img, label="selfie")
        except ValueError as e:
            # No face detected — treat as a capture-quality issue (bad
            # angle/lighting/crop), not a hard failure.
            return {"decision": "needs_retake", "rejection_reason": str(e)}

        similarity = engines.face_engine.cosine_similarity(id_embedding, selfie_embedding)
        verdict = engines.face_engine.verdict(similarity)

        result = {
            "id_face_embedding": id_embedding,
            "selfie_face_embedding": selfie_embedding,
            "face_similarity": similarity,
            "face_verdict": verdict,
        }
        if verdict == "fail":
            # Design decision: similarity below FACE_MATCH_LOW is treated
            # as a retake case, not an automatic fraud reject — manual_review
            # is reserved solely for the ambiguous band between LOW and HIGH.
            # Tradeoff worth knowing: this means a genuine impostor (a face
            # that truly doesn't match) also just gets sent to "try again"
            # rather than being blocked. If that's a concern, consider
            # rate-limiting retake attempts per application at the API layer
            # — this graph itself has no attempt counter anymore.
            result["decision"] = "needs_retake"
            result["rejection_reason"] = "Face similarity below threshold — needs retake"
        return result
    return _node


# ---------------------------------------------------------------------------
# 9. LLM consolidation agent — STUB, prompt/Groq wiring not finalized yet
# ---------------------------------------------------------------------------

def llm_consolidation_node(engines: KYCEngines):
    def _node(state: KYCState) -> dict:
        consolidated = engines.consolidation_agent.consolidate_all(state.get("enhanced_reads", {}))
        merged = {**state.get("extracted_fields", {}), **{k: v for k, v in consolidated.items() if v}}
        return {"extracted_fields": merged}
    return _node

# ---------------------------------------------------------------------------
# 10. Final policy decision
# ---------------------------------------------------------------------------

def final_decision_node(state: KYCState) -> dict:
    if state.get("decision") in ("rejected", "needs_retake"):
        return {}  # already decided by an earlier short-circuit

    if state.get("is_duplicate"):
        return {"decision": "rejected", "rejection_reason": "Duplicate application detected"}

    # expiry_valid is never None here — expiry_check_node now short-circuits
    # unreadable dates straight to needs_retake, so by this point the date
    # was read and IS valid (an invalid/expired date is a "rejected"
    # short-circuit too). The only thing left to decide is the face match.
    verdict = state.get("face_verdict")
    if verdict == "pass":
        return {"decision": "approved"}
    if verdict == "review":
        # The one and only manual_review trigger: a face similarity in the
        # ambiguous band (FACE_MATCH_LOW <= sim < FACE_MATCH_HIGH) — a
        # genuine judgment call, not a quality problem.
        return {"decision": "manual_review", "rejection_reason": "Face similarity needs human review"}

    # Defensive fallback only — "fail" verdicts already exit via
    # needs_retake in face_match_node, so this shouldn't be reachable.
    return {"decision": "manual_review", "rejection_reason": "Unhandled case — defaulting to review"}


# ---------------------------------------------------------------------------
# Email / mobile handoff — the single exit point for every image-quality
# problem (no card detected, too blurry, no face detected, unreadable
# expiry date, or a face similarity below FACE_MATCH_LOW). Phase 3 redesign
# collapsed the old bounded local-retry loop and the separate handoff node
# into just this: first failure routes straight here, no local looping.
# Also reused for the final-confirmation "No" path in the frontend
# (Phase 5) — the frontend calls this same graph re-entry point rather
# than needing a second implementation of the same fallback.
# ---------------------------------------------------------------------------

def email_handoff_node(state: KYCState) -> dict:
    """
    TODO (Phase 4/5): actually send the magic-link email (SMTP/SendGrid or
    similar) with a single-use, time-limited token for the mobile
    continuation session — that needs the FastAPI backend + DB to exist.
    For now this just marks the terminal state so the frontend knows to
    prompt for an email address and trigger the real send.
    """
    return {
        "decision": "mobile_handoff_required",
        "requires_mobile_handoff": True,
        "rejection_reason": state.get("rejection_reason") or "Image quality insufficient — mobile handoff offered",
    }