# """
# Assembles the KYC LangGraph workflow.

# Order (Phase 3 design — see project notes for reasoning):
#   ID front capture -> ID back capture -> OCR consensus -> expiry check
#   (short-circuit reject if expired, or needs_retake if unreadable) ->
#   duplicate check -> face match (selfie already captured upstream, e.g.
#   by the UI liveness flow — this graph is tested with saved images per
#   the Phase 3 plan, not live webcam) -> LLM consolidation -> final
#   decision.

# Cheaper/less-invasive checks run before expensive ones (expiry before
# face match), matching the fail-fast ordering agreed in design review.

# Quality-issue handling (Phase 3 redesign): there used to be three
# separate "something's wrong" exits — a bounded local retry loop, a
# missing-values manual_review, and a separate mobile/email handoff. They
# collapsed into two, since in practice they were almost all the same root
# cause (poor image quality):

#   - needs_retake  -> no card detected, all captured frames too blurry
#                       (see config.BLUR_THRESHOLD), no face detected, an
#                       unreadable expiry date, or a face similarity below
#                       FACE_MATCH_LOW. ALL of these route straight to
#                       email_handoff on first failure — no local self-loop
#                       retry anymore (magic-link mobile continuation;
#                       actual email sending is a Phase 4/5 TODO, see
#                       nodes.py).
#   - manual_review -> reserved for exactly one case: a face similarity in
#                       the ambiguous band (FACE_MATCH_LOW <= sim < HIGH).
#                       That's a genuine judgment call, not a quality
#                       problem — see face_match_node / final_decision_node.

# `rejected` (expired ID, duplicate application) is separate from both of
# the above — those are real policy rejections on a document that WAS read
# successfully, not quality issues.
# """
# from langgraph.graph import StateGraph, END

# from src2.graph.state import KYCState
# from src2.graph.nodes import (
#     KYCEngines,
#     capture_id_front_node,
#     capture_id_back_node,
#     ocr_consensus_node,
#     expiry_check_node,
#     duplicate_check_node,
#     face_match_node,
#     llm_consolidation_node,
#     final_decision_node,
#     email_handoff_node,
# )
# from src2.graph.edges import (
#     route_after_front_capture,
#     route_after_back_capture,
#     route_after_expiry_check,
#     route_after_duplicate_check,
#     route_after_face_match,
# )


# def build_kyc_graph(engines: KYCEngines):
#     graph = StateGraph(KYCState)

#     graph.add_node("capture_id_front", capture_id_front_node(engines))
#     graph.add_node("capture_id_back", capture_id_back_node(engines))
#     graph.add_node("ocr_consensus", ocr_consensus_node)
#     graph.add_node("expiry_check", expiry_check_node)
#     graph.add_node("duplicate_check", duplicate_check_node)
#     graph.add_node("face_match", face_match_node(engines))
#     graph.add_node("llm_consolidation", llm_consolidation_node(engines))
#     graph.add_node("final_decision", final_decision_node)
#     graph.add_node("email_handoff", email_handoff_node)

#     graph.set_entry_point("capture_id_front")

#     graph.add_conditional_edges(
#         "capture_id_front",
#         route_after_front_capture,
#         {
#             "capture_id_back": "capture_id_back",
#             "email_handoff": "email_handoff",  # needs_retake: straight to mobile handoff, no local self-loop
#         },
#     )
#     graph.add_conditional_edges(
#         "capture_id_back",
#         route_after_back_capture,
#         {
#             "ocr_consensus": "ocr_consensus",
#             "email_handoff": "email_handoff",  # needs_retake: straight to mobile handoff, no local self-loop
#         },
#     )
#     graph.add_edge("ocr_consensus", "expiry_check")
#     graph.add_conditional_edges(
#         "expiry_check",
#         route_after_expiry_check,
#         {
#             "duplicate_check": "duplicate_check",
#             "email_handoff": "email_handoff",  # needs_retake: unreadable expiry date
#             "end": END,  # rejected: expired
#         },
#     )
#     graph.add_conditional_edges(
#         "duplicate_check",
#         route_after_duplicate_check,
#         {"face_match": "face_match", "final_decision": "final_decision"},
#     )
#     graph.add_conditional_edges(
#         "face_match",
#         route_after_face_match,
#         {
#             "llm_consolidation": "llm_consolidation",
#             "email_handoff": "email_handoff",  # needs_retake: no face detected, or similarity below FACE_MATCH_LOW
#         },
#     )
#     graph.add_edge("llm_consolidation", "final_decision")
#     graph.add_edge("final_decision", END)
#     graph.add_edge("email_handoff", END)

#     return graph.compile()


# if __name__ == "__main__":
#     # Quick manual test against saved images — fill in real paths before running.
#     engines = KYCEngines.load()
#     kyc_graph = build_kyc_graph(engines)

#     initial_state = {
#         "id_front_frame_paths": ["D:/Gam3a/for_me/verfication_system/test/front.jpeg"],
#         "id_back_frame_paths": ["D:/Gam3a/for_me/verfication_system/test/back.jpeg"],
#         "selfie_frame_path": "D:/Gam3a/for_me/verfication_system/test/me.jpeg",
#         "confirmation_retry_count": 0,
#     }

#     final_state = kyc_graph.invoke(initial_state)
#     print("Decision:", final_state.get("decision"))
#     print("Extracted fields:", final_state.get("extracted_fields"))
#     print("Face similarity:", final_state.get("face_similarity"))
#     print("Rejection reason:", final_state.get("rejection_reason"))







    
######for testing with multiple frames





# """
# Assembles the KYC LangGraph workflow.

# Order (Phase 3 design — see project notes for reasoning):
#   ID front capture -> ID back capture -> OCR consensus -> expiry check
#   (short-circuit reject if expired, or needs_retake if unreadable) ->
#   duplicate check -> face match (selfie already captured upstream, e.g.
#   by the UI liveness flow — this graph is tested with saved images per
#   the Phase 3 plan, not live webcam) -> LLM consolidation -> final
#   decision.

# Cheaper/less-invasive checks run before expensive ones (expiry before
# face match), matching the fail-fast ordering agreed in design review.

# Quality-issue handling: needs_retake (no card detected, all captured
# frames too blurry, no face detected, an unreadable expiry date, or a
# face similarity below FACE_MATCH_LOW) routes straight to email_handoff
# — that node/edge MUST be registered below or the router raises
# KeyError('email_handoff') the first time any of those conditions fire,
# which is exactly what was happening before this fix.
# """
# import glob
# from langgraph.graph import StateGraph, END

# from src2.graph.state import KYCState
# from src2.graph.nodes import (
#     KYCEngines,
#     capture_id_front_node,
#     capture_id_back_node,
#     ocr_consensus_node,
#     expiry_check_node,
#     duplicate_check_node,
#     face_match_node,
#     llm_consolidation_node,
#     final_decision_node,
#     email_handoff_node,          # <-- was missing
# )
# from src2.graph.edges import (
#     route_after_front_capture,
#     route_after_back_capture,
#     route_after_expiry_check,
#     route_after_duplicate_check,
#     route_after_face_match,
# )


# def build_kyc_graph(engines: KYCEngines):
#     graph = StateGraph(KYCState)

#     graph.add_node("capture_id_front", capture_id_front_node(engines))
#     graph.add_node("capture_id_back", capture_id_back_node(engines))
#     graph.add_node("ocr_consensus", ocr_consensus_node)
#     graph.add_node("expiry_check", expiry_check_node)
#     graph.add_node("duplicate_check", duplicate_check_node)
#     graph.add_node("face_match", face_match_node(engines))
#     graph.add_node("llm_consolidation", llm_consolidation_node(engines))
#     graph.add_node("final_decision", final_decision_node)
#     graph.add_node("email_handoff", email_handoff_node)   # <-- was missing

#     graph.set_entry_point("capture_id_front")

#     graph.add_conditional_edges(
#         "capture_id_front",
#         route_after_front_capture,
#         {
#             "capture_id_back": "capture_id_back",
#             "email_handoff": "email_handoff",   # <-- was missing
#         },
#     )
#     graph.add_conditional_edges(
#         "capture_id_back",
#         route_after_back_capture,
#         {
#             "ocr_consensus": "ocr_consensus",
#             "email_handoff": "email_handoff",   # <-- was missing
#         },
#     )
#     graph.add_edge("ocr_consensus", "expiry_check")
#     graph.add_conditional_edges(
#         "expiry_check",
#         route_after_expiry_check,
#         {
#             "duplicate_check": "duplicate_check",
#             "email_handoff": "email_handoff",   # <-- was missing, this is the one that crashed on you
#             "end": END,
#         },
#     )
#     graph.add_conditional_edges(
#         "duplicate_check",
#         route_after_duplicate_check,
#         {"face_match": "face_match", "final_decision": "final_decision"},
#     )
#     graph.add_conditional_edges(
#         "face_match",
#         route_after_face_match,
#         {
#             "llm_consolidation": "llm_consolidation",
#             "email_handoff": "email_handoff",   # <-- was missing
#         },
#     )
#     graph.add_edge("llm_consolidation", "final_decision")
#     graph.add_edge("final_decision", END)
#     graph.add_edge("email_handoff", END)        # <-- was missing

#     return graph.compile()


# if __name__ == "__main__":
#     # Quick manual test against saved images — fill in real paths before running.
#     engines = KYCEngines.load()
#     kyc_graph = build_kyc_graph(engines)

#     # 1. Define folder paths for front and back frames
#     front_folder = "D:/Gam3a/for_me/verfication_system/test/front2"
#     back_folder = "D:/Gam3a/for_me/verfication_system/test/back"

#     # 2. Use glob to fetch all images inside those folders
#     front_frames = glob.glob(f"{front_folder}/*.jpeg") + glob.glob(f"{front_folder}/*.jpg")
#     back_frames = glob.glob(f"{back_folder}/*.jpeg") + glob.glob(f"{back_folder}/*.jpg")

#     if not front_frames:
#         print(f"WARNING: No images found in {front_folder}")
#     if not back_frames:
#         print(f"WARNING: No images found in {back_folder}")

#     initial_state = {
#         "id_front_frame_paths": front_frames,
#         "id_back_frame_paths": back_frames,
#         "selfie_frame_path": "D:/Gam3a/for_me/verfication_system/test/me2.png",
#         "confirmation_retry_count": 0,
#     }

#     print(f"Starting graph with {len(front_frames)} front frames and {len(back_frames)} back frames...")
#     final_state = kyc_graph.invoke(initial_state)

#     print("\n--- Final Results ---")
#     print("Decision:", final_state.get("decision"))
#     print("Extracted fields:", final_state.get("extracted_fields"))
#     print("Face similarity:", final_state.get("face_similarity"))

#     if "rejection_reason" in final_state:
#         print("Rejection Reason:", final_state.get("rejection_reason"))




#######################for phase 4,5###############################
###################################################################
"""
Assembles the KYC LangGraph workflow.
"""
from langgraph.graph import StateGraph, END
from src2.graph.state import KYCState
from src2.graph.nodes import (
    KYCEngines, liveness_check_node, capture_id_front_node,
    capture_id_back_node, ocr_consensus_node, expiry_check_node,
    duplicate_check_node, face_match_node, llm_consolidation_node,
    final_decision_node, email_handoff_node,
)
from src2.graph.edges import (
    route_after_liveness_check, route_after_front_capture,
    route_after_back_capture, route_after_expiry_check,
    route_after_duplicate_check, route_after_face_match,
)

def build_kyc_graph(engines: KYCEngines, session_factory):
    graph = StateGraph(KYCState)
    graph.add_node("liveness_check", liveness_check_node(engines))
    graph.add_node("capture_id_front", capture_id_front_node(engines))
    graph.add_node("capture_id_back", capture_id_back_node(engines))
    graph.add_node("face_match", face_match_node(engines)) 
    graph.add_node("ocr_consensus", ocr_consensus_node)
    graph.add_node("expiry_check", expiry_check_node)
    graph.add_node("duplicate_check", duplicate_check_node(session_factory))
    graph.add_node("llm_consolidation", llm_consolidation_node(engines))
    graph.add_node("final_decision", final_decision_node)
    graph.add_node("email_handoff", email_handoff_node)

    graph.set_entry_point("liveness_check")
    
    graph.add_conditional_edges("liveness_check", route_after_liveness_check, {"capture_id_front": "capture_id_front", "email_handoff": "email_handoff"})
    graph.add_conditional_edges("capture_id_front", route_after_front_capture, {"capture_id_back": "capture_id_back", "email_handoff": "email_handoff"})
    

    graph.add_conditional_edges("capture_id_back", route_after_back_capture, {"face_match": "face_match", "email_handoff": "email_handoff"})
    

    graph.add_conditional_edges("face_match", route_after_face_match, {"ocr_consensus": "ocr_consensus", "email_handoff": "email_handoff"})
    graph.add_edge("ocr_consensus", "expiry_check")
    
    graph.add_conditional_edges("expiry_check", route_after_expiry_check, {"duplicate_check": "duplicate_check", "email_handoff": "email_handoff", "end": END})
    graph.add_conditional_edges("duplicate_check", route_after_duplicate_check, {"llm_consolidation": "llm_consolidation", "final_decision": "final_decision"})
    
    graph.add_edge("llm_consolidation", "final_decision")
    graph.add_edge("final_decision", END)
    graph.add_edge("email_hando_off" if False else "email_handoff", END)
    
    return graph.compile()