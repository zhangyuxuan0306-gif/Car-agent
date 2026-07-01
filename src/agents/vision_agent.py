"""Vision Agent — 相同角色并行识别局部地标"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from src.agents.schemas import VisionResult
from src.perception.custom_yolo_detector import CustomYoloDetector, YoloDetection

logger = logging.getLogger(__name__)


class VisionAgent:
    """视觉识别智能体（多个实例共享同一检测器，并行处理不同切片）"""

    def __init__(self, agent_id: str, detector: CustomYoloDetector):
        self.agent_id = agent_id
        self.detector = detector

    def recognize(
        self,
        tile_id: str,
        crop: np.ndarray,
        full_frame: np.ndarray,
        tile_center: Tuple[int, int],
        tile_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> VisionResult:
        """对局部图像块做 YOLO 识别，失败时回退到全图最近检测框"""
        tx1, ty1, _, _ = tile_bbox
        dets = self.detector.detect(crop)
        if dets:
            best = max(dets, key=lambda d: d.confidence)
            bx1, by1, bx2, by2 = best.bbox
            full_bbox = (bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1)
            return VisionResult(
                agent_id=self.agent_id,
                tile_id=tile_id,
                building_name=best.label,
                confidence=best.confidence,
                bbox=full_bbox,
            )

        # 回退：在全图中以切片中心找最近建筑
        cx, cy = tile_center
        full_dets = self.detector.detect(full_frame)
        if not full_dets:
            return VisionResult(
                agent_id=self.agent_id,
                tile_id=tile_id,
                building_name="未识别建筑",
                confidence=0.0,
            )

        best = self.detector.pick_at_point(full_frame, cx, cy, dets=full_dets)
        if best:
            return VisionResult(
                agent_id=self.agent_id,
                tile_id=tile_id,
                building_name=best.label,
                confidence=best.confidence * 0.85,
                bbox=best.bbox,
            )

        return VisionResult(
            agent_id=self.agent_id,
            tile_id=tile_id,
            building_name=full_dets[0].label,
            confidence=full_dets[0].confidence * 0.7,
            bbox=full_dets[0].bbox,
        )
