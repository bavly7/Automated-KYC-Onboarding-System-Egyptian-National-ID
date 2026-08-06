# Automated KYC Onboarding System — Phase 0 Design Doc

## 1. Objective

Build an end-to-end identity verification pipeline that automates customer onboarding
for a bank/fintech use case: extract identity data from an Egyptian national ID,
confirm the applicant is a live person, verify they match the ID photo, and issue an
automated approve / reject / manual-review decision.

## 2. Scope

**In scope**
- Egyptian national ID card — front and back
- Live webcam capture flow (not static file upload)
- OCR field extraction
- Face verification (selfie vs. ID photo)
- Motion-based liveness check
- Automated decision routing (approve / reject / manual review)

**Out of scope (v1)**
- Passports or any non-Egyptian ID format
- Full active anti-spoofing / depth-based liveness detection
- Specific street address extraction (see 3.2)
- Multi-language OCR beyond Arabic/English as printed on the card

## 3. Data to extract

### 3.1 Required fields (front)
| Field | Notes |
|---|---|
| First name | |
| Last name | |
| National ID number | 14-digit — validate format/checksum, not just raw OCR text |
| Region / governorate | Coarse address |
| Expiry date | Drives the valid/expired check |

### 3.2 Optional / deferred
- Specific street address (back) — deferred to future work. Harder to OCR reliably
  (multi-line, less standardized) and lower value for demonstrating the core
  identity + expiry verification use case.

### 3.3 Back-of-card fields
- Barcode / additional printed fields — TBD once back-layout OCR is prototyped in Phase 2.

## 4. Capture flow

1. Webcam session starts.
2. Liveness check: user performs a movement challenge (turn head / raise hand),
   detected via face + hand landmarks, before any frame is accepted as "the" face frame.
3. On successful liveness confirmation, capture one clear frontal face frame.
4. Prompt: capture front of ID.
5. Prompt: capture back of ID.

## 5. Verification logic

### 5.1 Document validity
- If OCR-extracted expiry date is in the past → **reject immediately**, message:
  "Please upload a valid ID." Face verification does not run on an already-invalid document.

### 5.2 Face verification
- Compare captured selfie frame against the face photo extracted from the ID.
- **Known challenge**: appearance drift since the ID photo was issued (beard, glasses,
  lighting) can lower match confidence for genuinely valid users.
- **Design decision**: use a three-tier threshold rather than binary match/no-match,
  so appearance drift routes to manual review instead of a false reject:
  - High similarity → pass
  - Low similarity → fail
  - Borderline similarity → manual review

### 5.3 Decision outcomes
| Outcome | Conditions |
|---|---|
| **Approve** | ID not expired + OCR fields extracted with sufficient confidence + liveness check passed + face match score above the "high similarity" threshold |
| **Manual review** | All of the above pass, but face match score falls in the borderline band (appearance drift case), OR OCR confidence is low without being unreadable |
| **Reject** | ID expired, OR liveness check failed, OR face match score below the "low similarity" threshold, OR document unreadable after retry |

*(Exact threshold values TBD in Phase 2, once face-match scores are measured on a real
validation set.)*

## 6. Open questions for later phases
- Exact similarity score thresholds for the three-tier face match (Phase 2)
- OCR confidence threshold for triggering the re-capture loop vs. manual review (Phase 2)
- Whether back-of-card barcode data will be used for anything beyond field cross-check (Phase 2)
