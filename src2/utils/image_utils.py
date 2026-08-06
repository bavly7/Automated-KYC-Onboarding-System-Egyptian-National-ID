"""
Frame-quality and multi-frame consensus helpers, used to turn 10 captured
frames per ID side into a single trustworthy value per field.
"""
from collections import Counter

import cv2
import numpy as np


def blur_score(img: np.ndarray) -> float:
    """Higher = sharper. Variance of the Laplacian — cheap, no model needed."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def select_best_frames(frames: list, top_n: int = 10) -> list:
    """Pre-filter to the sharpest N frames before running OCR on all of them."""
    scored = [(blur_score(f), f) for f in frames]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in scored[:top_n]]


def consensus_value(values: list):
    """Majority vote across frame-level OCR reads for one field. Ignores empty/None."""
    counts = Counter(v for v in values if v)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def consensus_per_field(reads_per_field: dict) -> dict:
    """
    reads_per_field: {field_name: [value_frame1, value_frame2, ...]}
    Returns {field_name: consensus_value}
    """
    return {field: consensus_value(values) for field, values in reads_per_field.items()}
