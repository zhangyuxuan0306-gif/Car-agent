"""眼动追踪模块 - 基于 MediaPipe Face Landmarker"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from src.utils.model_assets import ensure_models

logger = logging.getLogger(__name__)

# 虹膜关键点索引
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP = 1


@dataclass
class GazeResult:
    gaze_point: Tuple[float, float]
    gaze_point_px: Tuple[int, int]
    confidence: float
    face_detected: bool
    head_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class GazeTracker:
    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smoothing_window: int = 5
    _landmarker: Optional[FaceLandmarker] = field(default=None, repr=False)
    _gaze_history: Deque = field(default_factory=deque, repr=False)
    _timestamp_ms: int = field(default=0, repr=False)

    def __post_init__(self):
        self._gaze_history = deque(maxlen=self.smoothing_window)

    def _init(self):
        if self._landmarker is not None:
            return
        paths = ensure_models()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=paths["face_landmarker.task"]),
            running_mode=RunningMode.VIDEO,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        logger.info("MediaPipe Face Landmarker 已初始化")

    def _estimate_head_pose(self, landmarks, w: int, h: int) -> Tuple[float, float, float]:
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
        ], dtype=np.float64)

        def pt(idx):
            lm = landmarks[idx]
            return lm.x * w, lm.y * h

        image_points = np.array([
            pt(NOSE_TIP), pt(152), pt(33), pt(263), pt(61), pt(291),
        ], dtype=np.float64)

        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)

        success, rvec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return float(angles[0]), float(angles[1]), float(angles[2])

    def _iris_center(self, landmarks, indices: List[int], w: int, h: int) -> Tuple[float, float]:
        xs = [landmarks[i].x * w for i in indices]
        ys = [landmarks[i].y * h for i in indices]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def track(self, frame: np.ndarray) -> GazeResult:
        self._init()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            return GazeResult(
                gaze_point=(0.5, 0.5), gaze_point_px=(w // 2, h // 2),
                confidence=0.0, face_detected=False,
            )

        landmarks = result.face_landmarks[0]
        n_landmarks = len(landmarks)

        if n_landmarks > 478:
            left_iris = self._iris_center(landmarks, LEFT_IRIS, w, h)
            right_iris = self._iris_center(landmarks, RIGHT_IRIS, w, h)
            gaze_x = (left_iris[0] + right_iris[0]) / 2
            gaze_y = (left_iris[1] + right_iris[1]) / 2
        else:
            # 无虹膜点时用眼中心
            gaze_x = (landmarks[33].x + landmarks[263].x) / 2 * w
            gaze_y = (landmarks[33].y + landmarks[263].y) / 2 * h

        pitch, yaw, roll = self._estimate_head_pose(landmarks, w, h)
        gaze_x = np.clip(gaze_x + yaw * 0.15 * w, 0, w)
        gaze_y = np.clip(gaze_y + pitch * 0.15 * h, 0, h)

        norm_x, norm_y = gaze_x / w, gaze_y / h
        self._gaze_history.append((norm_x, norm_y))
        avg_x = sum(p[0] for p in self._gaze_history) / len(self._gaze_history)
        avg_y = sum(p[1] for p in self._gaze_history) / len(self._gaze_history)
        confidence = min(1.0, len(self._gaze_history) / self.smoothing_window)

        return GazeResult(
            gaze_point=(avg_x, avg_y),
            gaze_point_px=(int(avg_x * w), int(avg_y * h)),
            confidence=confidence,
            face_detected=True,
            head_pose=(pitch, yaw, roll),
        )

    def draw_gaze(self, frame: np.ndarray, gaze: GazeResult) -> np.ndarray:
        vis = frame.copy()
        if not gaze.face_detected:
            cv2.putText(vis, "No Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return vis
        px, py = gaze.gaze_point_px
        color = (0, 255, 255)
        cv2.circle(vis, (px, py), 20, color, 2)
        cv2.line(vis, (px - 25, py), (px + 25, py), color, 2)
        cv2.line(vis, (px, py - 25), (px, py + 25), color, 2)
        cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)
        info = f"Gaze: ({gaze.gaze_point[0]:.2f}, {gaze.gaze_point[1]:.2f}) conf={gaze.confidence:.2f}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return vis

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
