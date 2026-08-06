"""
LangGraph state schema. This is also the shape of the final structured
output — designed to map cleanly onto the Postgres schema in Phase 4
without needing a DB connection during Phase 3 graph testing.
"""
from typing import TypedDict, Optional, List


class KYCState(TypedDict, total=False):
    # ---- inputs (saved image paths for Phase 3 testing; webcam-fed later) ----
    id_front_frame_paths: List[str]
    id_back_frame_paths: List[str]
    selfie_frame_path: Optional[str]

    # ---- liveness (Phase 4/5, Option C - Hybrid) ----
    liveness_frame_paths: List[str]
    liveness_instructions: List[str]
    liveness_passed: Optional[bool]
    liveness_pass_count: Optional[int]

    # ---- intermediate artifacts ----
    front_card_crop: object          # numpy array
    back_card_crop: object           # numpy array
    front_field_boxes: dict
    back_field_boxes: dict

    # ---- OCR consensus ----
    # Enhanced-only: raw OCR was tested head-to-head against enhanced and
    # lost (see project notes), so the pipeline no longer tracks raw reads.
    enhanced_reads: dict               # {field: [values across frames]}
    extracted_fields: dict             # final consensus per field, post LLM consolidation

    # ---- validation guardrails ----
    expiry_valid: Optional[bool]
    is_duplicate: Optional[bool]

    # ---- face verification ----
    id_face_embedding: object
    selfie_face_embedding: object
    face_similarity: Optional[float]
    face_verdict: Optional[str]       # "pass" | "review" | "fail"

    # ---- retry / guardrail counters ----
    # capture_retry_count removed (Phase 3 redesign): capture/face-match
    # failures no longer loop the same node — they route straight to
    # needs_retake -> email_handoff instead, so there's nothing to count.
    confirmation_retry_count: int  # unrelated: frontend "confirm extracted data" retry, Phase 5
    requires_mobile_handoff: bool

    # ---- terminal outcome ----
    # "needs_retake" is a transient routing signal (never the final decision
    # in practice) — email_handoff_node always overwrites it with
    # "mobile_handoff_required" once reached.
    decision: Optional[str]  # "approved" | "manual_review" | "rejected" | "needs_retake" | "mobile_handoff_required"
    rejection_reason: Optional[str]
