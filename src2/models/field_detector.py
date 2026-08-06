"""
Field-region detection — locates ID/First_Name/Last_Name/Add1/ExpDate boxes
on an already-cropped card image.
"""
import numpy as np
from ultralytics import YOLO

from src2 import config


class FieldDetector:
    def __init__(
        self,
        weights_path: str = config.FIELD_MODEL_PATH,
        target_classes=config.TARGET_CLASSES,
    ):
        self.model = YOLO(weights_path)
        self.target_classes = set(target_classes)

    def detect_fields(self, card_crop: np.ndarray, conf: float = config.FIELD_DETECT_CONF) -> dict:
        """
        card_crop: BGR numpy array (output of CardDetector.detect_and_crop)
        Returns {field_name: (x1, y1, x2, y2)} restricted to target_classes.
        """
        result = self.model.predict(source=card_crop, conf=conf, verbose=False)[0]

        field_boxes = {
            self.model.names[int(box.cls)]: tuple(int(v) for v in box.xyxy[0])
            for box in result.boxes
            if self.model.names[int(box.cls)] in self.target_classes
        }
        return field_boxes
