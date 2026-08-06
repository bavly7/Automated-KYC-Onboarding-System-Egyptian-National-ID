# Automated KYC Onboarding System — Egyptian National ID

An end-to-end identity verification pipeline: a user captures their Egyptian
national ID (front + back) and a liveness-checked selfie through a browser,
and a LangGraph-orchestrated agent extracts their identity data, verifies
they're a live person who matches the ID photo, and issues an automated
**approve / manual review / reject / mobile handoff** decision.

Built as an applied CV/ML portfolio project — YOLO (card + field detection),
PaddleOCR (Arabic field extraction), InsightFace (ArcFace face verification),
MediaPipe (liveness), a Groq-powered LLM consolidation agent, LangGraph
(orchestration), and a FastAPI + PostgreSQL backend.

## What it does

1. **Liveness check** — the user performs 3 randomly-chosen movement
   challenges (e.g. "turn your head right," "raise your left hand") on
   camera. Runs as the *first* node in the graph — fail here, and nothing
   else runs.
2. **ID capture** — 10 frames each of the front and back of the card.
   A YOLO model crops the card boundary, a second YOLO model locates each
   field (name, ID number, address, expiry date), and PaddleOCR reads each
   field per frame. A majority vote across the 10 frames produces the
   consensus value per field.
3. **Expiry validation** — an expired or unreadable date short-circuits
   the pipeline before the more expensive steps below ever run.
4. **Duplicate check** — rejects a second application using a national ID
   number that's already been approved.
5. **Face verification** — InsightFace embeddings compare the live selfie
   against the face photo cropped from the ID, using a **three-tier**
   threshold rather than a binary match/no-match, specifically to handle
   appearance drift (e.g. a beard or glasses not present in the ID photo)
   without over-rejecting genuine users:
   - high similarity → pass
   - low similarity → needs retake (treated as a quality issue, not a
     block — see *Known Limitations*)
   - the ambiguous middle band → **manual review**
6. **LLM consolidation** — a Groq-powered agent normalizes the OCR
   consensus per field (e.g. trims a two-word OCR read down to just the
   first name), constrained so it can only select/lightly normalize among
   values that actually appeared in the OCR output — never invent one.
7. **Final decision** — `approved`, `manual_review`, `rejected`, or
   `mobile_handoff_required`.

Any image-quality problem anywhere in the pipeline (no card detected, every
captured frame too blurry, no face detected, an unreadable expiry date, or
low face similarity) routes to the **same** fallback: a magic-link email
offering to continue the session on a phone instead.

## Architecture

```mermaid
flowchart TD
    START([Start]) --> LC[liveness_check]
    LC -->|OK| CF[capture_id_front]
    LC -->|needs_retake| EH[email_handoff]

    CF -->|OK| CB[capture_id_back]
    CF -->|needs_retake| EH

    CB -->|OK| OC[ocr_consensus]
    CB -->|needs_retake| EH

    OC --> EC[expiry_check]

    EC -->|valid| DC[duplicate_check]
    EC -->|needs_retake| EH
    EC -->|rejected: expired| END1([End])

    DC -->|unique| FM[face_match]
    DC -->|is_duplicate| FD[final_decision]

    FM -->|OK| LLM[llm_consolidation]
    FM -->|needs_retake: no face / low similarity| EH

    LLM --> FD
    FD --> END2([End])
    EH --> END3([End])
```

`needs_retake` is a fail-fast pattern used throughout: any step that can't
get usable data goes straight to `email_handoff` on the *first* failure —
there's no bounded local retry loop. `manual_review` is reserved for
exactly one case (an ambiguous face-match score), since that's a genuine
human judgment call rather than a quality problem.

## Tech stack

| Layer | Tech |
|---|---|
| Card / field detection | YOLO11 (Ultralytics), fine-tuned on a Roboflow Egyptian-ID dataset |
| OCR | PaddleOCR (Arabic), CLAHE + unsharp-mask enhancement |
| Face verification | InsightFace (ArcFace embeddings), cosine similarity |
| Liveness | MediaPipe Tasks API (FaceLandmarker, PoseLandmarker) |
| LLM consolidation | Groq |
| Orchestration | LangGraph |
| Backend | FastAPI, SQLAlchemy |
| Database | PostgreSQL (SQLite supported for local testing) |
| Frontend | Vanilla JS + HTML, browser camera capture |

## Setup

**1. Install dependencies**
```bash
pip install ultralytics paddleocr paddlepaddle insightface onnxruntime \
            mediapipe opencv-python-headless numpy langgraph groq \
            python-dotenv fastapi uvicorn sqlalchemy psycopg2-binary \
            pydantic[email]
```

**2. Model weights** — place these under a `Models/` folder (paths are
configurable via env vars, see `config.py`):
- `best_card_detector.pt` — card boundary YOLO model
- `best_labels1.pt` — field detection YOLO model
- `face_landmarker.task`, `pose_landmarker.task` — MediaPipe Tasks models

**3. `.env` file** at the project root:
```dotenv
GROQ_API_KEY=your_groq_key

# Postgres (recommended) or SQLite (fine for local testing)
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/kyc_db
# DATABASE_URL=sqlite:///./kyc.db

# Must be an address your PHONE can actually reach — see Known Limitations
FRONTEND_BASE_URL=http://<your-lan-ip>:5500/index.html

# Optional — omit to just print the magic link to the console instead of emailing it
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
EMAIL_FROM=no-reply@kyc-demo.local
```

**4. Run the backend**
```bash
uvicorn src2.backend.main:app --reload --host 0.0.0.0 --port 8000
```

**5. Serve the frontend** (needs to be served over HTTP, not opened as a
`file://` path, since the resume-from-email-link flow reads URL query
params):
```bash
cd frontend && python -m http.server 5500
```
Update `API_BASE` at the top of `index.html` to match wherever the backend
is actually reachable from your browser/phone (a LAN IP, not `localhost`,
if you want to test the mobile-handoff flow from a real phone).

## API reference

| Endpoint | Purpose |
|---|---|
| `POST /sessions` | Create a session; returns `session_id` + 3 randomly-chosen liveness instructions |
| `POST /sessions/{id}/liveness` | Submit the 3 liveness frames; fails fast before any ID capture starts |
| `POST /sessions/{id}/upload` | Submit 10 front + 10 back + 1 selfie frame; runs the graph in the background |
| `GET /sessions/{id}/status` | Poll for the result (`pending` → `processing` → `done`/`error`) |
| `POST /sessions/{id}/handoff` | Request a retry — either reset the current session or email a mobile continuation link |
| `GET /sessions/{id}/resume` | Consume a single-use magic-link token to resume capture on another device |

## Design decisions worth knowing

- **Enhanced-only OCR.** Raw vs. CLAHE-enhanced OCR were tested head-to-head
  with a majority-vote consensus experiment against ground truth; enhanced
  won, so the pipeline runs enhanced OCR only (raw is still available in
  `OCREngine.extract_raw` if you want to re-run that comparison on more cards).
- **`needs_retake` has no rate limit at the graph level.** A face similarity
  below the low threshold is treated as "please retry" rather than an
  automatic block — intentional, to avoid over-rejecting genuine users on a
  bad photo, but it does mean nothing in this graph currently stops
  someone from retrying indefinitely. See *Known Limitations*.
- **`duplicate_check` only compares national ID numbers** against previously
  *approved* sessions. A face-embedding similarity search across applicants
  (catching someone re-applying under a different identity) is real
  fraud-detection value but out of scope for now — would need a vector
  index (e.g. pgvector) and its own threshold calibration.
- **All numeric thresholds are placeholders**: `BLUR_THRESHOLD`,
  `FACE_MATCH_HIGH`/`FACE_MATCH_LOW`, `LIVENESS_PASS_THRESHOLD`,
  `OCR_OVERLAP_THRESH` were all set from limited hand-testing, not a
  calibrated validation set. Treat them as a starting point.

## Known limitations / open issues

Documenting these here rather than papering over them:

- **Liveness can be double-checked with mismatched frame ordering.**
  `POST /sessions/{id}/liveness` saves the 3 frames in the exact order the
  frontend sent them (matching the instruction order). But because
  `liveness_check_node` also runs again as the graph's entry node during
  `/upload` (same design — liveness is the first pipeline step), and that
  second run rebuilds the frame list via `Path.glob("*")` on the saved
  files, the frames can come back in a **different order** than they were
  captured in (`glob` doesn't guarantee capture order, and the saved
  filenames are UUID-prefixed). That mismatched ordering against
  `liveness_instructions` explains inconsistent pass/fail behavior between
  the immediate check and the final result. **Fix**: persist the
  frame-to-instruction pairing explicitly (e.g. save as
  `0_turn_right.jpg`, `1_raise_hand.jpg`, or store the ordered path list on
  the session row) instead of relying on either capture order or
  `glob()`'s directory order.
- **Mobile handoff links won't load on a phone if `FRONTEND_BASE_URL` is
  left at its default** (`http://localhost:5500/...`) — `localhost` on the
  *sender's* machine means nothing to the phone receiving the link. Set it
  to a LAN IP reachable from the phone (same pattern as `API_BASE` in
  `index.html`), and note both devices need to be on the same network for
  this demo setup to work at all — there's no public hosting involved.
- **SQLite vs. PostgreSQL** — `DATABASE_URL` is currently pointed at SQLite
  for local testing; the code itself is DB-agnostic via SQLAlchemy, so
  switching is just an env var change, but this hasn't been re-tested
  against Postgres since the switch (see *Incomplete Phases* below).
- **`llm_agent.py`'s `GROQ_MODEL` is currently set to
  `llama-3.3-70b-versatile`**, which Groq has deprecated (confirmed via
  Groq's own deprecation notice) — this will likely fail at runtime.
  Switch to `openai/gpt-oss-120b`, Groq's recommended replacement.
- **`config.TARGET_CLASSES` currently includes `"Add2"`**, while the Phase
  0 design doc and `field_detector.py`'s own docstring both still say
  `"Add1"` — worth double-checking which field was actually intended
  before this becomes a silent mismatch between the docs and the code.
- **No document-authenticity / anti-spoofing check** on the ID card itself
  (a printed photocopy or a photo-of-a-photo would currently pass) — this
  is the document-side equivalent of face liveness and was scoped out from
  the start, not forgotten.
- **Thresholds and the field schema are all Egyptian-national-ID-specific**
  by design (see Phase 0 doc) — no passport or other-country support.

## Project phase status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scope & design doc | Done |
| 1 | Dataset collection & prep | Done |
| 2 | Model prototyping (OCR, face embeddings, liveness — standalone) | Done |
| 3 | Agent orchestration (LangGraph) | Done |
| 4 | Backend (FastAPI + PostgreSQL) | Built, but running on SQLite — Postgres swap not yet re-tested |
| 5 | Frontend capture flow | Done |
| 6 | **Containerization (Docker)** | **Not started** |
| 7 | **End-to-end evaluation** (measured accuracy, not just a working demo) | **Not done** — a working end-to-end demo run confirms the pipeline functions, but no field-level OCR accuracy, face-match precision/recall, or false-accept/false-reject rate has been measured against a labeled set. Every threshold in `config.py` is still an unvalidated placeholder as a direct result. |
| 8 | Documentation & demo | README done; demo video not yet recorded |

## Future improvement ideas

Not yet implemented — capturing the reasoning now so the "why" isn't lost
before they're picked up:

**1. Replace the fixed enhancement pipeline with a vision-agent-based image
adjustment step.** Right now `enhance_card()` applies the same fixed
CLAHE + unsharp-mask transform to every image regardless of what's actually
wrong with it. Given access to a vision-capable agent/tool, the idea is to
have it inspect each captured frame and apply *whatever* correction that
specific image needs (deglare, deblur, exposure correction, etc.) rather
than one static filter applied uniformly — in principle more robust across
the range of real-world lighting/camera conditions the fixed pipeline
can't adapt to. Worth prototyping against the same raw-vs-enhanced
consensus methodology already used elsewhere in this project, rather than
assuming it's strictly better without measurement.

**2. Cross-check extracted names against an identity-data provider (e.g.
LexisNexis) as an additional duplicate/fraud signal.** Current
`duplicate_check_node` only matches on the exact national ID number. The
idea: after name extraction, query a service like LexisNexis to check
whether the extracted name is associated with signals suggesting it may
not uniquely resolve to one person — and if so, route to `manual_review`
rather than auto-approving, on the reasoning that a name collision (same
name, different real person) is a distinct risk from an exact ID-number
duplicate and isn't caught by the current check at all. Would need real
scoping around what "found" means from that API (a name match alone is
extremely common and not inherently suspicious) before this could avoid
flooding manual review with false positives.

## Next steps

- Fix the liveness frame-ordering bug and re-test
- Point `FRONTEND_BASE_URL` at a real LAN-reachable address and re-test the
  mobile handoff end to end
- Switch `DATABASE_URL` to Postgres and re-test
- Build a small labeled validation set and run Phase 7 for real: report OCR
  field accuracy, face-match precision/recall, false-accept/false-reject rate
- Containerize with Docker Compose (backend + Postgres + model inference)
- Rate-limit `needs_retake`/handoff attempts at the API layer

