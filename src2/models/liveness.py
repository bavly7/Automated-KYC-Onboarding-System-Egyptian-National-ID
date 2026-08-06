"""
Liveness check: verifies a captured frame satisfies a given movement
instruction (head turn / hand raise), using MediaPipe's Tasks API
(FaceLandmarker / PoseLandmarker) — NOT the legacy `mp.solutions` API,
which is currently broken on Python 3.12+ (see project notes).

This module only *checks* a single frame against an instruction. Driving
the webcam loop and picking random instructions is a UI-layer concern
(Colab JS bridge today, frontend capture flow in Phase 5) and stays out
of this module on purpose — keep this testable on saved frames too.
"""
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from src2 import config


class LivenessChecker:
    def __init__(
        self,
        face_model_path: str = config.FACE_LANDMARKER_PATH,
        pose_model_path: str = config.POSE_LANDMARKER_PATH,
    ):
        self.face_landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=face_model_path),
                running_mode=vision.RunningMode.IMAGE,
            )
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=pose_model_path),
                running_mode=vision.RunningMode.IMAGE,
            )
        )

    def check_instruction(self, img_rgb, instruction: str) -> bool:
        """
        img_rgb: RGB numpy array (already converted from BGR by the caller)
        instruction: one of config.LIVENESS_INSTRUCTIONS
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        if "head" in instruction:
            result = self.face_landmarker.detect(mp_image)
            if not result.face_landmarks:
                print(f"[liveness debug] '{instruction}': NO FACE DETECTED in frame")
                return False
            lm = result.face_landmarks[0]
            nose_x = lm[1].x
            mid_x = (lm[454].x + lm[234].x) / 2
            offset = nose_x - mid_x
            # NOTE: sign convention here matches a mirrored (selfie) camera
            # feed — flip if your capture pipeline doesn't mirror the image.
            passed = offset < -0.03 if "right" in instruction else offset > 0.03
            print(f"[liveness debug] '{instruction}': offset={offset:.4f} (need <-0.03 or >0.03) -> {'PASS' if passed else 'FAIL'}")
            return passed

        if "hand" in instruction:
            result = self.pose_landmarker.detect(mp_image)
            if not result.pose_landmarks:
                print(f"[liveness debug] '{instruction}': NO POSE DETECTED in frame")
                return False
            lm = result.pose_landmarks[0]
            # mirrored selfie view: user's right hand appears on frame-left
            if "right" in instruction:
                wrist, shoulder = lm[16], lm[12]
            else:
                wrist, shoulder = lm[15], lm[11]
            delta = wrist.y - shoulder.y
            passed = delta < -0.05
            print(f"[liveness debug] '{instruction}': wrist.y-shoulder.y={delta:.4f} (need <-0.05) -> {'PASS' if passed else 'FAIL'}")
            return passed

        return False