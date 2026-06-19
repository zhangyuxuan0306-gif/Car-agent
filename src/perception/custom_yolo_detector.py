"""自定义 YOLOv8 建筑检测"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_YOLO_ROOT = os.path.join(ROOT, "models", "yolo")


@dataclass
class YoloDetection:
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    label: str
    confidence: float
    class_id: int

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def contains(self, px: int, py: int) -> bool:
        x1, y1, x2, y2 = self.bbox
        return x1 <= px <= x2 and y1 <= py <= y2

    def contains_point(self, x: float, y: float) -> bool:
        return self.contains(int(x), int(y))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class CustomYoloDetector:
    """自训练 YOLOv8 检测器"""

    def __init__(
        self,
        yolo_root: str = DEFAULT_YOLO_ROOT,
        weights: str = "weights/best_epoch_weights.pth",
        classes_file: str = "model_data/voc_classes.txt",
        confidence: float = 0.4,
        nms_iou: float = 0.3,
        phi: str = "s",
        input_shape: Tuple[int, int] = (640, 640),
    ):
        self.yolo_root = yolo_root
        self.weights_path = os.path.join(yolo_root, weights)
        self.classes_path = os.path.join(yolo_root, classes_file)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.phi = phi
        self.input_shape = input_shape
        self._yolo = None
        self.class_names: List[str] = []

    def _load(self):
        if self._yolo is not None:
            return
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"权重不存在: {self.weights_path}")
        if self.yolo_root not in sys.path:
            sys.path.insert(0, self.yolo_root)

        from yolo import YOLO

        logger.info("加载自定义 YOLO: %s", self.weights_path)
        self._yolo = YOLO(
            model_path=self.weights_path,
            classes_path=self.classes_path,
            confidence=self.confidence,
            nms_iou=self.nms_iou,
            phi=self.phi,
            input_shape=list(self.input_shape),
            cuda=torch.cuda.is_available(),
        )
        self.class_names = list(self._yolo.class_names)
        logger.info("检测类别: %s", self.class_names)

    def detect(self, frame_bgr: np.ndarray) -> List[YoloDetection]:
        """检测一帧，返回所有建筑框"""
        self._load()
        from utils.utils import cvtColor, preprocess_input, resize_image

        image = cvtColor(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        image_shape = np.array(image.size[::-1])  # h, w

        image_data = resize_image(
            image, (self.input_shape[1], self.input_shape[0]), self._yolo.letterbox_image
        )
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self._yolo.cuda:
                images = images.cuda()
            outputs = self._yolo.net(images)
            outputs = self._yolo.bbox_util.decode_box(outputs)
            results = self._yolo.bbox_util.non_max_suppression(
                outputs, self._yolo.num_classes, self.input_shape,
                image_shape, self._yolo.letterbox_image,
                conf_thres=self.confidence, nms_thres=self.nms_iou,
            )

        if results[0] is None:
            return []

        dets: List[YoloDetection] = []
        for row in results[0]:
            top, left, bottom, right = row[:4]
            conf = float(row[4])
            cls_id = int(row[5])
            x1 = max(0, int(left))
            y1 = max(0, int(top))
            x2 = min(frame_bgr.shape[1] - 1, int(right))
            y2 = min(frame_bgr.shape[0] - 1, int(bottom))
            if x2 <= x1 or y2 <= y1:
                continue
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            dets.append(YoloDetection(
                bbox=(x1, y1, x2, y2),
                label=label,
                confidence=conf,
                class_id=cls_id,
            ))
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets

    def pick_at_point(
        self,
        frame_bgr: np.ndarray,
        px: int,
        py: int,
        dets: Optional[List[YoloDetection]] = None,
    ) -> Optional[YoloDetection]:
        """点击位置选框：优先包含点击点的最高置信度框，否则最近框"""
        if dets is None:
            dets = self.detect(frame_bgr)
        if not dets:
            logger.warning("当前帧未检测到建筑")
            return None

        inside = [d for d in dets if d.contains(px, py)]
        if inside:
            best = max(inside, key=lambda d: d.confidence)
            logger.info("命中检测框: %s (%.2f)", best.label, best.confidence)
            return best

        # 最近中心
        best, best_d = None, float("inf")
        for d in dets:
            cx, cy = d.center
            dist = (cx - px) ** 2 + (cy - py) ** 2
            if dist < best_d:
                best_d, best = dist, d
        if best:
            logger.info("最近检测框: %s (%.2f)", best.label, best.confidence)
        return best

    @staticmethod
    def describe(label: str, brief_service=None) -> str:
        if brief_service is not None:
            return brief_service.describe(label)
        from src.multimodal.building_brief import BuildingBriefService
        return BuildingBriefService().describe(label)

    @staticmethod
    def find_at_point(dets: List[YoloDetection], px: int, py: int) -> int:
        inside = [i for i, d in enumerate(dets) if d.contains(px, py)]
        if inside:
            return max(inside, key=lambda i: dets[i].confidence)
        best_i, best_d = -1, float("inf")
        for i, d in enumerate(dets):
            cx, cy = d.center
            dist = (cx - px) ** 2 + (cy - py) ** 2
            if dist < best_d:
                best_d, best_i = dist, i
        return best_i

    @staticmethod
    def crop_target(frame: np.ndarray, det: YoloDetection) -> np.ndarray:
        x1, y1, x2, y2 = det.bbox
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        dets: List[YoloDetection],
        highlight_idx: int = -1,
        point_px: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        vis = frame.copy()
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d.bbox
            color = (0, 0, 255) if i == highlight_idx else (0, 200, 0)
            thick = 3 if i == highlight_idx else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
            cv2.putText(
                vis, f"{d.label} {d.confidence:.2f}",
                (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        if point_px:
            cv2.circle(vis, point_px, 10, (0, 0, 255), -1)
        return vis
