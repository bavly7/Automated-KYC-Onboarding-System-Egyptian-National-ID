"""
Face embedding extraction + comparison, using InsightFace's built-in face
detector + alignment (no separate face-detection model needed, and no full
background segmentation — just detect + align, per project decision).
"""
import numpy as np
from insightface.app import FaceAnalysis

from src2 import config


class FaceEngine:
    def __init__(self, providers=("CPUExecutionProvider",)):
        self.app = FaceAnalysis(name="buffalo_l", providers=list(providers))
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, image: np.ndarray, label: str = "") -> np.ndarray:
        faces = self.app.get(image)
        if len(faces) == 0:
            raise ValueError(f"No face detected in {label or 'image'}")
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
        return faces[0].normed_embedding

    @staticmethod
    def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
        return float(np.dot(e1, e2))  # already L2-normalized

    @staticmethod
    def verdict(similarity: float) -> str:
        """
        Three-tier decision (Phase 0 design — handles appearance drift like
        beard/glasses without hard-rejecting genuine users).
        Returns one of: "pass", "review", "fail".
        """
        if similarity >= config.FACE_MATCH_HIGH:
            return "pass"
        if similarity >= config.FACE_MATCH_LOW:
            return "review"
        return "fail"
