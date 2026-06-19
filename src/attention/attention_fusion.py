"""注意力融合模块 - 融合眼动、手势与视觉检测推断关注建筑目标"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from src.perception.gaze_tracker import GazeResult
from src.perception.gesture_recognizer import GestureResult, GestureType

logger = logging.getLogger(__name__)


@dataclass
class AttentionScore:
    """单个检测目标的注意力得分"""
    detection: Any
    detection_idx: int
    total_score: float
    gaze_score: float
    gesture_score: float
    detection_score: float
    is_confirmed: bool = False


@dataclass
class AttentionResult:
    """注意力融合结果"""
    scores: List[AttentionScore]
    best_target: Optional[AttentionScore] = None
    gaze_point: Tuple[float, float] = (0.5, 0.5)
    gesture_active: bool = False
    frame_count: int = 0


@dataclass
class AttentionFusion:
    """多模态注意力融合引擎"""

    gaze_weight: float = 0.5
    gesture_weight: float = 0.35
    detection_weight: float = 0.15
    gaze_hit_radius: float = 0.35
    confirm_frames: int = 3
    frame_width: int = 1920
    frame_height: int = 1080
    _confirm_history: Dict[int, Deque[bool]] = field(default_factory=dict, repr=False)
    _frame_count: int = field(default=0, repr=False)

    def _gaze_score(
        self,
        gaze: GazeResult,
        det: Detection,
        w: int,
        h: int,
    ) -> float:
        """计算注视对目标的得分"""
        if not gaze.face_detected:
            return 0.0

        gx, gy = gaze.gaze_point
        # 归一化 bbox
        x1, y1, x2, y2 = det.bbox
        nx1, ny1 = x1 / w, y1 / h
        nx2, ny2 = x2 / w, y2 / h
        cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2

        # 注视点在 bbox 内得满分
        if nx1 <= gx <= nx2 and ny1 <= gy <= ny2:
            return 1.0 * gaze.confidence

        # 距离衰减
        dist = math.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        if dist > self.gaze_hit_radius:
            return 0.0
        return (1.0 - dist / self.gaze_hit_radius) * gaze.confidence

    def _gesture_score(
        self,
        gesture: GestureResult,
        det: Detection,
        w: int,
        h: int,
    ) -> float:
        """计算手势对目标的得分"""
        if gesture.primary_gesture != GestureType.POINTING:
            if gesture.primary_gesture == GestureType.PINCH:
                # 捏合手势：强化当前最高注视目标
                return 0.5
            return 0.0

        if gesture.pointing_target is None:
            return 0.0

        px, py = gesture.pointing_target
        x1, y1, x2, y2 = det.bbox
        nx1, ny1 = x1 / w, y1 / h
        nx2, ny2 = x2 / w, y2 / h

        if nx1 <= px <= nx2 and ny1 <= py <= ny2:
            return 1.0

        cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
        dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        if dist > self.gaze_hit_radius:
            return 0.0
        return 1.0 - dist / self.gaze_hit_radius

    def _detection_score(self, det: Detection, all_dets: List[Detection]) -> float:
        """检测置信度与面积综合得分"""
        if not all_dets:
            return 0.0
        max_area = max(d.area for d in all_dets)
        area_norm = det.area / max_area if max_area > 0 else 0
        return 0.6 * det.confidence + 0.4 * area_norm

    def fuse(
        self,
        detections: List[Any],
        gaze: GazeResult,
        gesture: GestureResult,
        frame_shape: Tuple[int, int],
    ) -> AttentionResult:
        """融合多模态信号，推断关注目标"""
        h, w = frame_shape[:2]
        self._frame_count += 1

        if not detections:
            return AttentionResult(
                scores=[],
                best_target=None,
                gaze_point=gaze.gaze_point,
                gesture_active=gesture.primary_gesture != GestureType.NONE,
                frame_count=self._frame_count,
            )

        scores: List[AttentionScore] = []
        for i, det in enumerate(detections):
            gs = self._gaze_score(gaze, det, w, h)
            ges = self._gesture_score(gesture, det, w, h)
            ds = self._detection_score(det, detections)

            total = (
                self.gaze_weight * gs
                + self.gesture_weight * ges
                + self.detection_weight * ds
            )

            # 多帧确认
            if i not in self._confirm_history:
                self._confirm_history[i] = deque(maxlen=self.confirm_frames)
            self._confirm_history[i].append(total > 0.4)
            confirmed = sum(self._confirm_history[i]) >= self.confirm_frames

            scores.append(AttentionScore(
                detection=det,
                detection_idx=i,
                total_score=total,
                gaze_score=gs,
                gesture_score=ges,
                detection_score=ds,
                is_confirmed=confirmed,
            ))

        scores.sort(key=lambda s: s.total_score, reverse=True)
        best = scores[0] if scores and scores[0].total_score > 0.15 else None

        if best and not best.is_confirmed and self._frame_count < self.confirm_frames:
            # 前几帧降低置信度
            pass

        return AttentionResult(
            scores=scores,
            best_target=best,
            gaze_point=gaze.gaze_point,
            gesture_active=gesture.primary_gesture != GestureType.NONE,
            frame_count=self._frame_count,
        )

    def reset(self):
        """重置多帧确认历史"""
        self._confirm_history.clear()
        self._frame_count = 0
