from src2.graph.state import KYCState


def route_after_liveness_check(state: KYCState) -> str:
    if state.get("decision") == "needs_retake":
        return "email_handoff"
    return "capture_id_front"


def route_after_front_capture(state: KYCState) -> str:
    if state.get("decision") == "needs_retake":
        return "email_handoff"
    return "capture_id_back"


def route_after_back_capture(state: KYCState) -> str:
    if state.get("decision") == "needs_retake":
        return "email_handoff"

    return "face_match"


def route_after_face_match(state: KYCState) -> str:
    decision = state.get("decision")

    if decision == "needs_retake":
        return "email_handoff"

    return "ocr_consensus"


def route_after_expiry_check(state: KYCState) -> str:
    decision = state.get("decision")
    
    # ID is expired (Terminal state)
    if decision == "rejected":
        return "end"
        
    # Unreadable dates on the first attempt (Routes to email_handoff)
    if decision == "needs_retake":
        return "email_handoff"
        
    # Unreadable dates on a mobile retry (Terminal state -> Human Queue)
    if decision == "manual_review":
        return "end" 
        
    # Expiry is valid, proceed to the next automated check
    return "duplicate_check"


def route_after_duplicate_check(state: KYCState) -> str:
    if state.get("is_duplicate"):
        return "final_decision"
    return "llm_consolidation"