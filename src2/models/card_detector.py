"""
Card boundary detection — finds and crops the ID card out of a full photo.
Wraps the notebook `detect_and_crop_card` function as a reusable class.
"""
import cv2
from ultralytics import YOLO

from src2 import config


class CardDetector:
    def __init__(self, weights_path: str = config.CARD_MODEL_PATH):
        self.model = YOLO(weights_path)

    def detect_and_crop(self, image_path: str, conf: float = config.CARD_DETECT_CONF):
        """
        Returns the cropped card region (BGR numpy array), or None if no
        card was detected — caller should trigger the re-capture flow.
        """
        result = self.model.predict(source=image_path, conf=conf, verbose=False)[0]

        if len(result.boxes) == 0:
            return None

        best_box = max(result.boxes, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = [int(v) for v in best_box.xyxy[0]]

        img = cv2.imread(image_path)
        return img[y1:y2, x1:x2]
