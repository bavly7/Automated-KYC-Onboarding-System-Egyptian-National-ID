"""
Central configuration: model weight paths, target fields, and thresholds.
Keep every "magic number" here so nodes/ and models/ never hardcode values.
"""
import os

from dotenv import load_dotenv

load_dotenv(override=True)  # .env always wins over a stray OS-level env var of the same name

# ---- Model weight paths -----------------------------------------------
# Point these at your Models/ folder. Override via env vars in production.
CARD_MODEL_PATH = os.environ.get("CARD_MODEL_PATH", "Models/best_card_detector.pt")
FIELD_MODEL_PATH = os.environ.get("FIELD_MODEL_PATH", "Models/best_labels1.pt")
FACE_LANDMARKER_PATH = os.environ.get("FACE_LANDMARKER_PATH", "Models/face_landmarker.task")
POSE_LANDMARKER_PATH = os.environ.get("POSE_LANDMARKER_PATH", "Models/pose_landmarker.task")

# ---- Field schema (Phase 0 decision) -----------------------------------
TARGET_CLASSES = {"ID", "First_Name", "Last_Name", "Add2", "ExpDate"}

# ---- Detection / OCR thresholds ----------------------------------------
CARD_DETECT_CONF = 0.25
FIELD_DETECT_CONF = 0.25
OCR_OVERLAP_THRESH = 0.30
OCR_MARGIN_PX = 8

# ---- Frame sharpness gate (Phase 3 decision) ---------------------------
# blur_score() (variance of Laplacian) ranks captured frames against EACH
# OTHER to pick the sharpest of the batch — it never checked whether even
# the sharpest one was actually good enough. This threshold is what makes
# "too blurry" an actual retake trigger: if the sharpest frame in the batch
# still scores below this, the whole side is treated as capture failure
# (routes to needs_retake -> email/mobile handoff) instead of silently
# proceeding with a bad-quality crop.
# PLACEHOLDER — calibrate against real sharp/blurry sample frames (this
# value is resolution- and lighting-dependent) before trusting it.
BLUR_THRESHOLD = 60.0

# ---- Face match three-tier thresholds (Phase 0 decision) --------------
# similarity >= FACE_MATCH_HIGH        -> pass
# FACE_MATCH_LOW <= similarity < HIGH  -> manual review (appearance drift)
# similarity < FACE_MATCH_LOW          -> needs_retake (treated as a quality
#                                          issue, not an automatic fraud
#                                          reject — see project notes)
# PLACEHOLDERS — calibrate against real same/different-person pairs before trusting these.
FACE_MATCH_HIGH = 0.45
FACE_MATCH_LOW = 0.25

# ---- Liveness ------------------------------------------------------------
# Phase 4/5 (Option C - Hybrid): frontend captures 3 "peak" frames, one per
# displayed instruction, instead of a real-time live video liveness check.
# The instructions for a session are chosen server-side in POST /sessions
# and handed back to the frontend so the graph (tested standalone on saved
# frames too) never has to reach into the DB to know what was asked.
LIVENESS_INSTRUCTIONS = [
    "Turn your head right",
    "Turn your head left",
    "Raise your right hand",
    "Raise your left hand",
]
LIVENESS_HOLD_FRAMES = 5
LIVENESS_NUM_CHALLENGES = 3  # was 2 — Phase 4/5 hybrid capture sends 3 frames
# Pass if at least this many of the 3 challenge frames satisfy their
# instruction. Not "all 3" on purpose — MediaPipe missing one landmark call
# on a single simulated "peak" frame (no real burst/temporal smoothing here)
# shouldn't fail an otherwise-live user. PLACEHOLDER — revisit after real testing.
LIVENESS_PASS_THRESHOLD = 2

# ---- Retry / safety guardrails -------------------------------------------
# MAX_CAPTURE_RETRIES removed (Phase 3 redesign): capture/face-match
# failures no longer bounce-retry the same node — they go straight to
# needs_retake -> email_handoff (mobile continuation) on first failure.
# See build_graph.py docstring for the full reasoning.
MAX_CONFIRMATION_RETRIES = 2  # unrelated: frontend "confirm extracted data" retry count, Phase 5
FRAMES_PER_SIDE = 10  # frames captured per ID side, used for majority-vote OCR consensus

# ---- Mobile handoff (Phase 5) ---------------------------------------------
# Base URL the handoff email links back to — this is the FRONTEND, not this
# API. Override in .env once you know where index.html is actually served
# from (a real HTTP server, not file://, since the resume flow needs to run
# JS against a URL with query params).
FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "http://localhost:5500/index.html")
HANDOFF_TOKEN_TTL_MINUTES = 30

# ---- Email (magic-link send for mobile handoff) ---------------------------
# All optional: if SMTP_HOST isn't set, email_utils.send_handoff_email()
# just prints the link to the console instead of sending — keeps the demo
# runnable without real SMTP credentials, per the Phase 4/5 TODO in nodes.py.
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "no-reply@kyc-demo.local")

# ---- Groq (LLM consolidation agent) --------------------------------------
# Read from .env — never hardcode the key here or commit it.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
