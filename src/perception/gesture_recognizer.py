"""手势识别模块 - 基于 MediaPipe Hand Landmarker"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from src.utils.model_assets import ensure_models

logger = logging.getLogger(__name__)


class GestureType(Enum):
    NONE = "none"
    POINTING = "pointing"
    OPEN_PALM = "open_palm"
    PINCH = "pinch"
    THUMBS_UP = "thumbs_up"


@dataclass
class HandResult:
    landmarks: List[Tuple[float, float, float]]
    gesture: GestureType
    pointing_direction: Tuple[float, float]
    confidence: float
    handedness: str


@dataclass
class GestureResult:
    hands: List[HandResult]
    primary_gesture: GestureType = GestureType.NONE
    pointing_target: Optional[Tuple[float, float]] = None


@dataclass
class GestureRecognizer:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    pointing_threshold: float = 160
    _landmarker: Optional[HandLandmarker] = field(default=None, repr=False)
    _timestamp_ms: int = field(default=0, repr=False)

    def _init(self):
        if self._landmarker is not None:
            return
        paths = ensure_models()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=paths["hand_landmarker.task"]),
            running_mode=RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        logger.info("MediaPipe Hand Landmarker 已初始化")

    @staticmethod
    def _angle(a, b, c) -> float:
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        if mag_ba * mag_bc < 1e-6:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))))

    def _classify_gesture(self, lms) -> Tuple[GestureType, float, Tuple[float, float]]:
        WRIST, THUMB_TIP, THUMB_IP = 0, 4, 3
        INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
        MIDDLE_TIP, MIDDLE_PIP = 12, 10
        RING_TIP, RING_PIP = 16, 14
        PINKY_TIP, PINKY_PIP = 20, 18

        def pt(i):
            return (lms[i].x, lms[i].y)

        index_ext = self._angle(pt(INDEX_MCP), pt(INDEX_PIP), pt(INDEX_TIP)) > self.pointing_threshold
        middle_ext = self._angle(pt(5), pt(MIDDLE_PIP), pt(MIDDLE_TIP)) > self.pointing_threshold
        ring_ext = self._angle(pt(13), pt(RING_PIP), pt(RING_TIP)) > self.pointing_threshold
        pinky_ext = self._angle(pt(17), pt(PINKY_PIP), pt(PINKY_TIP)) > self.pointing_threshold
        thumb_ext = pt(THUMB_TIP)[1] < pt(THUMB_IP)[1]

        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            direction = (pt(INDEX_TIP)[0] - pt(WRIST)[0], pt(INDEX_TIP)[1] - pt(WRIST)[1])
            mag = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
            if mag > 1e-6:
                direction = (direction[0] / mag, direction[1] / mag)
            return GestureType.POINTING, 0.9, direction

        pinch_dist = math.sqrt((pt(THUMB_TIP)[0] - pt(INDEX_TIP)[0]) ** 2 +
                               (pt(THUMB_TIP)[1] - pt(INDEX_TIP)[1]) ** 2)
        if pinch_dist < 0.05:
            return GestureType.PINCH, 0.85, (0.0, 0.0)

        if index_ext and middle_ext and ring_ext and pinky_ext:
            return GestureType.OPEN_PALM, 0.8, (0.0, 0.0)

        if thumb_ext and not index_ext and not middle_ext:
            return GestureType.THUMBS_UP, 0.75, (0.0, -1.0)

        return GestureType.NONE, 0.3, (0.0, 0.0)

    def recognize(self, frame: np.ndarray) -> GestureResult:
        self._init()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hand_results: List[HandResult] = []
        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                gesture, conf, direction = self._classify_gesture(hand_lms)
                handedness = "Unknown"
                if result.handedness and i < len(result.handedness):
                    handedness = result.handedness[i][0].category_name
                lms = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                hand_results.append(HandResult(
                    landmarks=lms, gesture=gesture,
                    pointing_direction=direction, confidence=conf,
                    handedness=handedness,
                ))

        primary = GestureType.NONE
        pointing_target = None
        pointing_hands = [h for h in hand_results if h.gesture == GestureType.POINTING]
        if pointing_hands:
            primary = GestureType.POINTING
            hand = pointing_hands[0]
            tip_x = hand.landmarks[8][0]
            tip_y = hand.landmarks[8][1]
            pointing_target = (
                float(np.clip(tip_x + hand.pointing_direction[0] * 0.3, 0, 1)),
                float(np.clip(tip_y + hand.pointing_direction[1] * 0.3, 0, 1)),
            )
        elif hand_results:
            primary = hand_results[0].gesture

        return GestureResult(hands=hand_results, primary_gesture=primary, pointing_target=pointing_target)

    def draw_gestures(self, frame: np.ndarray, gesture: GestureResult) -> np.ndarray:
        vis = frame.copy()
        h, w = frame.shape[:2]
        colors = {
            GestureType.POINTING: (0, 255, 0), GestureType.PINCH: (255, 0, 255),
            GestureType.OPEN_PALM: (255, 255, 0), GestureType.THUMBS_UP: (0, 165, 255),
            GestureType.NONE: (128, 128, 128),
        }
        for hand in gesture.hands:
            color = colors.get(hand.gesture, (128, 128, 128))
            for lm in hand.landmarks:
                cv2.circle(vis, (int(lm[0] * w), int(lm[1] * h)), 4, color, -1)
            if hand.gesture == GestureType.POINTING:
                tip = hand.landmarks[8]
                tx, ty = int(tip[0] * w), int(tip[1] * h)
                dx = int(hand.pointing_direction[0] * 80)
                dy = int(hand.pointing_direction[1] * 80)
                cv2.arrowedLine(vis, (tx, ty), (tx + dx, ty + dy), (0, 255, 0), 3, tipLength=0.3)

        if gesture.primary_gesture != GestureType.NONE:
            cv2.putText(vis, f"Gesture: {gesture.primary_gesture.value}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        colors.get(gesture.primary_gesture, (255, 255, 255)), 2)
        if gesture.pointing_target:
            cv2.circle(vis, (int(gesture.pointing_target[0] * w),
                             int(gesture.pointing_target[1] * h)), 15, (0, 255, 0), 2)
        return vis

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
