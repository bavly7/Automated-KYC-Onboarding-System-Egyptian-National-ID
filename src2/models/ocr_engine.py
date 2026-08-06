"""
OCR engine: PaddleOCR wrapper + enhancement + spatial matching of OCR text
lines to YOLO field boxes.

NOTE on preprocessing flags: run OCR on the *whole card crop*, not on tiny
per-field crops (that approach was tried and produced unusable output — see
project notes). Keep use_doc_orientation_classify / use_doc_unwarping /
use_textline_orientation OFF for this pipeline: the card is already a clean,
roughly-flat crop from CardDetector, not a photographed full page, so that
preprocessing does more harm than good here (confirmed empirically).
"""
import cv2
import numpy as np
from paddleocr import PaddleOCR

from src2 import config


def enhance_card(img: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement + unsharp mask. Returns a new image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    return sharpened


def box_overlap_ratio(ocr_poly, field_box, margin: int = config.OCR_MARGIN_PX) -> float:
    """
    Fraction of the OCR text box's area that falls inside the YOLO field
    box, expanded by `margin` px on each side to tolerate small detection
    boundary imprecision (needed especially for lower-confidence classes
    like First_Name — see project notes).
    """
    poly = np.array(ocr_poly)
    ox1, oy1, ox2, oy2 = poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()

    fx1, fy1, fx2, fy2 = field_box
    fx1, fy1, fx2, fy2 = fx1 - margin, fy1 - margin, fx2 + margin, fy2 + margin

    ix1, iy1 = max(ox1, fx1), max(oy1, fy1)
    ix2, iy2 = min(ox2, fx2), min(oy2, fy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    ocr_area = (ox2 - ox1) * (oy2 - oy1)
    return inter / ocr_area if ocr_area > 0 else 0.0


def match_ocr_to_fields(field_boxes: dict, ocr_result, overlap_thresh: float = config.OCR_OVERLAP_THRESH) -> dict:
    """
    Assigns each OCR-detected text line to the field box it overlaps most.
    Returns {field_name: extracted_text}.
    """
    if not ocr_result or not ocr_result[0]:
        return {}

    rec_polys = ocr_result[0]["rec_polys"]
    rec_texts = ocr_result[0]["rec_texts"]

    extracted = {}
    for field_name, fbox in field_boxes.items():
        for poly, text in zip(rec_polys, rec_texts):
            if box_overlap_ratio(poly, fbox) >= overlap_thresh:
                extracted[field_name] = (extracted.get(field_name, "") + " " + text).strip()

    return extracted


class OCREngine:
    """Single PaddleOCR instance, reused across raw and enhanced calls."""

    def __init__(self, lang: str = "ar"):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
        )

    def extract_raw(self, card_crop: np.ndarray, field_boxes: dict) -> dict:
        result = self.ocr.ocr(card_crop)
        return match_ocr_to_fields(field_boxes, result)

    def extract_enhanced(self, card_crop: np.ndarray, field_boxes: dict) -> dict:
        enhanced = enhance_card(card_crop)
        result = self.ocr.ocr(enhanced)
        return match_ocr_to_fields(field_boxes, result)
