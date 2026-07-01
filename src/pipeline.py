"""智能座舱街景理解系统 - 核心流水线"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from src.attention.attention_fusion import AttentionFusion, AttentionResult
from src.interaction.qa_engine import QAEngine
from src.multimodal.building_brief import BuildingBriefService
from src.multimodal.knowledge_base import BuildingKnowledgeBase
from src.multimodal.llm_synthesizer import LLMAnswerSynthesizer
from src.multimodal.web_knowledge import web_searcher_from_config
from src.perception.custom_yolo_detector import CustomYoloDetector
from src.perception.gaze_tracker import GazeTracker, GazeResult
from src.perception.gesture_recognizer import GestureRecognizer, GestureResult, GestureType, HandResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    frame: Optional[np.ndarray] = None
    detections: list = field(default_factory=list)
    gaze: Optional[GazeResult] = None
    gesture: Optional[GestureResult] = None
    attention: Optional[AttentionResult] = None
    visualization: Optional[np.ndarray] = None
    target_crop: Optional[np.ndarray] = None
    building_description: str = ""
    building_identification: str = ""
    knowledge_context: str = ""
    processing_time_ms: float = 0.0
    target_label: str = ""


class CockpitScenePipeline:
    """智能座舱街景理解主流水线"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        root = os.path.dirname(os.path.abspath(config_path))
        det_cfg = self.config.get("custom_yolo", {})
        yolo_root = det_cfg.get("root", "models/yolo")
        if not os.path.isabs(yolo_root):
            yolo_root = os.path.join(root, yolo_root)
        self.detector = CustomYoloDetector(
            yolo_root=yolo_root,
            weights=det_cfg.get("weights", "weights/best_epoch_weights.pth"),
            classes_file=det_cfg.get("classes", "model_data/voc_classes.txt"),
            confidence=det_cfg.get("confidence", 0.25),
            nms_iou=det_cfg.get("nms_iou", 0.3),
            phi=det_cfg.get("phi", "s"),
        )

        gaze_cfg = self.config["gaze"]
        self.gaze_tracker = GazeTracker(
            max_num_faces=gaze_cfg["max_num_faces"],
            min_detection_confidence=gaze_cfg["min_detection_confidence"],
            min_tracking_confidence=gaze_cfg["min_tracking_confidence"],
            smoothing_window=gaze_cfg["smoothing_window"],
        )

        gesture_cfg = self.config["gesture"]
        self.gesture_recognizer = GestureRecognizer(
            max_num_hands=gesture_cfg["max_num_hands"],
            min_detection_confidence=gesture_cfg["min_detection_confidence"],
            min_tracking_confidence=gesture_cfg["min_tracking_confidence"],
            pointing_threshold=gesture_cfg["pointing_threshold"],
        )

        att_cfg = self.config["attention"]
        self.attention_fusion = AttentionFusion(
            gaze_weight=att_cfg["gaze_weight"],
            gesture_weight=att_cfg["gesture_weight"],
            detection_weight=att_cfg["detection_weight"],
            gaze_hit_radius=att_cfg["gaze_hit_radius"],
            confirm_frames=att_cfg["confirm_frames"],
        )

        kb_cfg = self.config["knowledge"]
        qa_cfg = self.config.get("qa", {})
        offline_qa = qa_cfg.get("offline", True)
        use_llm = qa_cfg.get("use_llm", False)
        use_web = kb_cfg.get("web_search", False)
        self.knowledge = BuildingKnowledgeBase(
            db_path=os.path.join(root, kb_cfg["db_path"])
            if not os.path.isabs(kb_cfg["db_path"])
            else kb_cfg["db_path"],
        )
        self.web_search = web_searcher_from_config(kb_cfg)
        llm_cfg = self.config.get("llm", {})
        device = self.config.get("system", {}).get("device", "cuda")
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        synthesizer = None
        if use_llm:
            synthesizer = LLMAnswerSynthesizer(
                model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
                device=device,
                max_new_tokens=llm_cfg.get("max_new_tokens", 128),
                temperature=llm_cfg.get("temperature", 0.3),
            )
        self.brief = BuildingBriefService(
            kb=self.knowledge,
            web=self.web_search,
            synthesizer=synthesizer,
            max_brief_len=kb_cfg.get("max_brief_len", 30),
            offline_qa=offline_qa,
            use_llm=use_llm,
            use_web=use_web,
        )
        self.qa_engine = QAEngine(brief=self.brief)

        self.state = PipelineState()

    def load_vlm(self):
        if self.brief.use_llm and self.brief.synthesizer:
            try:
                self.brief.preload_llm_async()
                return "✅ 大模型后台加载中"
            except Exception as e:
                return f"⚠️ 大模型加载失败（{e}）"
        if self.brief.use_web:
            return "✅ 本地问答 + 联网补全已就绪"
        return "✅ 纯本地问答已就绪（毫秒级响应）"

    def process_frame(
        self,
        frame: np.ndarray,
        point_px: Optional[Tuple[int, int]] = None,
    ) -> PipelineState:
        t0 = time.perf_counter()
        state = PipelineState()
        state.frame = frame
        h, w = frame.shape[:2]

        state.detections = self.detector.detect(frame)

        if point_px is not None:
            px, py = point_px
            state.gaze = GazeResult(
                gaze_point=(px / w, py / h),
                gaze_point_px=(px, py),
                confidence=1.0,
                face_detected=True,
            )
            state.gesture = GestureResult(
                hands=[HandResult(
                    landmarks=[(px / w, py / h, 0)] * 21,
                    gesture=GestureType.POINTING,
                    pointing_direction=(0.0, -0.3),
                    confidence=1.0,
                    handedness="Right",
                )],
                primary_gesture=GestureType.POINTING,
                pointing_target=(px / w, py / h),
            )
        else:
            state.gaze = self.gaze_tracker.track(frame)
            state.gesture = self.gesture_recognizer.recognize(frame)

        state.attention = self.attention_fusion.fuse(
            state.detections, state.gaze, state.gesture, frame.shape
        )

        vis = frame.copy()
        highlight_idx = -1
        if state.attention.best_target:
            highlight_idx = state.attention.best_target.detection_idx
            state.target_label = state.attention.best_target.detection.label
        elif point_px and state.detections:
            highlight_idx = CustomYoloDetector.find_at_point(
                state.detections, point_px[0], point_px[1]
            )

        vis = CustomYoloDetector.draw_detections(
            vis, state.detections, highlight_idx, point_px=point_px
        )
        if not point_px:
            vis = self.gaze_tracker.draw_gaze(vis, state.gaze)
            vis = self.gesture_recognizer.draw_gestures(vis, state.gesture)

        if state.attention and state.attention.best_target:
            bt = state.attention.best_target
            info = (
                f"Focus: {bt.detection.label} "
                f"score={bt.total_score:.2f}"
            )
            cv2.putText(vis, info, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        state.visualization = vis
        state.processing_time_ms = (time.perf_counter() - t0) * 1000
        self.state = state
        return state

    def analyze_target(self, frame: np.ndarray = None) -> Tuple[str, str]:
        if frame is None:
            frame = self.state.frame
        if frame is None:
            return "", "无可用图像"

        if not self.state.attention or not self.state.attention.best_target:
            return "", "请先注视或指向建筑。"

        target = self.state.attention.best_target.detection
        self.state.target_crop = CustomYoloDetector.crop_target(frame, target)
        identification = target.label
        description = self.brief.describe(target.label)
        kb_context = self.knowledge.get_context_for_building(target.label)

        self.state.building_identification = identification
        self.state.building_description = description
        self.state.knowledge_context = kb_context or description

        self.qa_engine.set_context(building_label=target.label, description=description)
        return identification, description

    def ask(self, question: str) -> str:
        return self.qa_engine.ask(question)

    def reset(self):
        self.attention_fusion.reset()
        self.qa_engine.clear_history()
        self.state = PipelineState()

    def cleanup(self):
        self.gaze_tracker.close()
        self.gesture_recognizer.close()
