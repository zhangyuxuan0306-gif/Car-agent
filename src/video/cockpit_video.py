"""车窗外景视频 — 自训练 YOLO + 本地/联网简介 + 终端问答"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import dlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.multimodal.building_brief import BuildingBriefService
from src.multimodal.knowledge_base import BuildingKnowledgeBase
from src.multimodal.llm_synthesizer import LLMAnswerSynthesizer
from src.multimodal.web_knowledge import WebKnowledgeSearcher
from src.perception.custom_yolo_detector import CustomYoloDetector, YoloDetection

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    active: bool = False
    click_point: Optional[Tuple[int, int]] = None
    center: Optional[Tuple[int, int]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    detection: Optional[YoloDetection] = None
    name: str = ""
    description: str = ""  # 点击时的短简介
    qa_answer: str = ""      # 最近一次问答完整回答
    frame_counter: int = 0
    frozen_frame: Optional[np.ndarray] = field(default=None, repr=False)


class CockpitVideoDemo:
    """自定义 YOLO 点击选楼 + 红点跟踪 + 透明遮罩"""

    def __init__(self, config: dict):
        video_cfg = config.get("video", {})
        yolo_cfg = config.get("custom_yolo", {})

        self.video_path = video_cfg.get("path", "data/videos/GuoMao.mp4")
        self.width = video_cfg.get("width", 1366)
        self.height = video_cfg.get("height", 768)
        self.fps = video_cfg.get("fps", 15)
        self.redetect_interval = video_cfg.get("redetect_interval", 10)

        yolo_root = yolo_cfg.get("root", os.path.join(ROOT, "models", "yolo"))
        self.detector = CustomYoloDetector(
            yolo_root=yolo_root,
            weights=yolo_cfg.get("weights", "weights/best_epoch_weights.pth"),
            classes_file=yolo_cfg.get("classes", "model_data/voc_classes.txt"),
            confidence=yolo_cfg.get("confidence", 0.4),
            nms_iou=yolo_cfg.get("nms_iou", 0.3),
            phi=yolo_cfg.get("phi", "s"),
        )
        kb_cfg = config.get("knowledge", {})
        llm_cfg = config.get("llm", {})
        device = config.get("system", {}).get("device", "cuda")
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        kb_path = kb_cfg.get("db_path", "data/knowledge/buildings.json")
        if not os.path.isabs(kb_path):
            kb_path = os.path.join(ROOT, kb_path)
        kb = BuildingKnowledgeBase(db_path=kb_path)
        qa_cfg = config.get("qa", {})
        offline_qa = qa_cfg.get("offline", True)
        use_llm = qa_cfg.get("use_llm", False)
        use_web = kb_cfg.get("web_search", False) and not offline_qa
        web = WebKnowledgeSearcher() if use_web else None
        synthesizer = None
        if use_llm:
            synthesizer = LLMAnswerSynthesizer(
                model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
                device=device,
                max_new_tokens=llm_cfg.get("max_new_tokens", 128),
                temperature=llm_cfg.get("temperature", 0.3),
            )
        self.brief = BuildingBriefService(
            kb=kb, web=web, synthesizer=synthesizer,
            max_brief_len=kb_cfg.get("max_brief_len", 30),
            offline_qa=offline_qa,
            use_llm=use_llm,
            use_web=use_web,
        )
        self.tracker = dlib.correlation_tracker()
        self.track = TrackState()
        self._window = "Smart Cockpit - Street View"
        self._all_dets: list[YoloDetection] = []
        self._running = False
        self._qa_building = ""          # 问答用，不随跟踪丢失而清空
        self._qa_busy = False
        self._qa_status = ""
        self._qa_progress = 0.0
        self._qa_queue: queue.Queue = queue.Queue()
        self._track_lock = threading.Lock()

    def _resolve_video(self) -> str:
        for p in [
            self.video_path,
            os.path.join(ROOT, self.video_path),
            os.path.join(ROOT, "data", "videos", "GuoMao.mp4"),
            os.path.join(ROOT, "GuoMao.mp4"),
        ]:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError("未找到演示视频")

    def _start_track(self, frame: np.ndarray, det: YoloDetection, click: Tuple[int, int]):
        h, w = frame.shape[:2]
        bbox = self._expand_bbox(det.bbox, w, h)
        x1, y1, x2, y2 = bbox
        cx, cy = int(det.center[0]), int(det.center[1])
        desc = self.brief.describe(det.label)

        self.track = TrackState(
            active=True,
            click_point=click,
            center=(cx, cy),
            bbox=bbox,
            detection=det,
            name=det.label,
            description=desc,
            frozen_frame=frame.copy(),
        )
        self._qa_building = det.label

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.tracker.start_track(rgb, dlib.rectangle(x1, y1, x2, y2))
        print(f"\n[识别] {det.label}  置信度 {det.confidence:.2f}")
        print(f"{desc}\n")

    def _refine_with_yolo(self, frame: np.ndarray):
        """周期性用 YOLO 校正跟踪框（问答进行中跳过，避免抢 GPU）"""
        if self._qa_busy or not self.track.active or not self.track.detection:
            return
        label = self.track.detection.label
        dets = self.detector.detect(frame)
        same = [d for d in dets if d.label == label]
        if not same:
            return
        # 选与当前 bbox IoU 最大的
        best = max(same, key=lambda d: self._iou(d.bbox, self.track.bbox))
        if self._iou(best.bbox, self.track.bbox) > 0.1:
            self.track.bbox = self._expand_bbox(best.bbox, frame.shape[1], frame.shape[0])
            self.track.center = (int(best.center[0]), int(best.center[1]))
            self.track.detection = best
            # YOLO 校正后重置跟踪器，避免 dlib 框逐渐缩小
            x1, y1, x2, y2 = self.track.bbox
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.tracker = dlib.correlation_tracker()
            self.tracker.start_track(rgb, dlib.rectangle(x1, y1, x2, y2))

    @staticmethod
    def _expand_bbox(bbox, w: int, h: int, margin: float = 0.04):
        """略微外扩检测框，缓解标注框偏紧"""
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * margin)
        pad_y = int(bh * margin)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w - 1, x2 + pad_x),
            min(h - 1, y2 + pad_y),
        )

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / (ua + 1e-6)

    def _update_track(self, frame: np.ndarray):
        if not self.track.active:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.tracker.update(rgb)
            pos = self.tracker.get_position()
            x1 = max(0, int(pos.left()))
            y1 = max(0, int(pos.top()))
            x2 = min(frame.shape[1] - 1, int(pos.right()))
            y2 = min(frame.shape[0] - 1, int(pos.bottom()))
            if x2 > x1 and y2 > y1:
                self.track.bbox = (x1, y1, x2, y2)
                self.track.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        except Exception as e:
            logger.debug("跟踪失败: %s", e)

        self.track.frame_counter += 1
        if self.track.frame_counter % self.redetect_interval == 0:
            self._refine_with_yolo(frame)

    def _render(self, frame: np.ndarray, show_all_boxes: bool = False) -> np.ndarray:
        vis = frame.copy().astype(np.float32)
        h, w = frame.shape[:2]

        # 可选：显示所有 YOLO 检测框（淡绿色）
        if show_all_boxes:
            for d in self._all_dets:
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(vis.astype(np.uint8), (x1, y1), (x2, y2), (0, 200, 0), 1)
            vis = vis.astype(np.uint8) if vis.dtype != np.uint8 else vis

        if not self.track.active or self.track.bbox is None:
            return vis.astype(np.uint8) if vis.dtype != np.uint8 else vis

        x1, y1, x2, y2 = self.track.bbox

        # 透明遮罩
        dark = vis * 0.22
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        m = mask[:, :, None].astype(np.float32)
        vis = vis * m + dark * (1.0 - m)
        vis = vis.astype(np.uint8)

        # 红框
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 中心红点
        if self.track.center:
            cx, cy = self.track.center
            cv2.circle(vis, (cx, cy), 12, (0, 0, 255), -1)
            cv2.circle(vis, (cx, cy), 18, (255, 255, 255), 2)

        # 名称 + 短简介（问答长文只在终端显示，避免画面卡顿）
        if self.track.name:
            text = self.track.name
            if self.track.description:
                text = f"{self.track.name}\n{self.track.description}"
            if self._qa_busy and self._qa_status:
                text += f"\n{self._qa_status}"
            elif self.track.qa_answer:
                # 显示问答摘要（首行）
                first_line = self.track.qa_answer.split("。")[0][:40]
                if first_line:
                    text += f"\n{first_line}…"
            tx = min(x2 + 8, w - 320)
            ty = max(y1, 10)
            vis = self._draw_text(vis, text, (tx, ty))

        if self._qa_busy:
            vis = self._draw_progress_bar(vis, self._qa_progress, self._qa_status)

        return vis

    def _draw_progress_bar(
        self, frame: np.ndarray, progress: float, label: str
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        progress = max(0.0, min(1.0, progress))
        bar_w, bar_h = min(420, w - 80), 20
        x0 = (w - bar_w) // 2
        y0 = h - 56
        pct = int(progress * 100)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - 2, y0 - 28), (x0 + bar_w + 2, y0 + bar_h + 8), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (80, 80, 80), -1)
        fill = max(0, int(bar_w * progress))
        if fill > 0:
            cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bar_h), (0, 200, 90), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)

        tip = label or "处理中"
        cv2.putText(
            frame, f"{tip}  {pct}%",
            (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return frame

    def _draw_text(self, frame: np.ndarray, text: str, anchor: Tuple[int, int]) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = None
        for fp in [
            os.path.join(ROOT, "models", "yolo", "model_data", "simhei.ttf"),
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]:
            if os.path.isfile(fp):
                font = ImageFont.truetype(fp, 22)
                break
        if font is None:
            font = ImageFont.load_default()
        x, y = anchor
        lines = textwrap.fill(text, width=22).split("\n")
        max_w = max((draw.textlength(ln, font=font) for ln in lines), default=0)
        draw.rectangle([x - 4, y - 4, x + int(max_w) + 10, y + len(lines) * 26 + 6], fill=(20, 20, 20))
        for line in lines:
            draw.text((x, y), line, font=font, fill=(255, 255, 255))
            y += 26
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _on_click(self, frame: np.ndarray, x: int, y: int):
        h, w = frame.shape[:2]
        x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        print(f"[点击] ({x}, {y}) — YOLO 检测中...")

        self._all_dets = self.detector.detect(frame)
        det = self.detector.pick_at_point(frame, x, y, dets=self._all_dets)

        if det is None:
            print("[提示] 未检测到建筑，请对准建筑物点击")
            return

        self._start_track(frame, det, (x, y))

    def _set_qa_progress(self, p: float, msg: str):
        self._qa_progress = p
        self._qa_status = msg

    def _qa_worker(self):
        """后台问答，不阻塞视频与跟踪"""
        while self._running:
            try:
                item = self._qa_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            building, question = item
            self._qa_busy = True
            self._set_qa_progress(0.05, "开始处理")

            def on_progress(p: float, msg: str):
                self._set_qa_progress(p, msg)

            t0 = time.time()
            try:
                ans = self.brief.answer(building, question, on_progress=on_progress)
                elapsed = time.time() - t0
                print(f"\n[问答 · {building} · {elapsed:.2f}s]\n{ans}\n", flush=True)
                with self._track_lock:
                    if self.track.active and self.track.name == building:
                        self.track.qa_answer = ans
            except Exception as e:
                logger.error("问答失败: %s", e)
                print(f"[问答] 失败: {e}", flush=True)
            finally:
                time.sleep(0.25)  # 让用户看到 100% 进度条
                self._qa_busy = False
                self._set_qa_progress(0.0, "")
                self._qa_queue.task_done()

    def _stdin_qa_loop(self):
        while self._running:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            q = (line or "").strip()
            if not q or not self._running:
                continue
            building = self._qa_building
            if not building:
                print("[问答] 请先点击选中一栋建筑", flush=True)
                continue
            if self._qa_busy:
                print("[问答] 上一条还在整理，请稍候…", flush=True)
                continue
            print(f"[问答] 已收到，处理中…", flush=True)
            self._qa_queue.put((building, q))

    def run(self):
        path = self._resolve_video()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开: {path}")

        dim = (self.width, self.height)
        delay = max(1, int(1000 / self.fps))
        show_all = False

        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window, 960, 540)

        current = [None]

        def mouse_cb(event, mx, my, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and current[0] is not None:
                self._on_click(current[0], mx, my)

        cv2.setMouseCallback(self._window, mouse_cb)

        print("\n=== 智能座舱 · 自训练 YOLO 建筑识别 ===")
        print("类别:", "、".join([
            "中国尊", "中央电视台总部大楼", "中央公园广场", "世界华商中心",
            "国贸三期B座", "国贸三期A座", "三星大厦",
        ]))
        print("左键点击建筑 | 终端问答（纯本地·秒回） | D R Q\n")

        self._running = True
        threading.Thread(target=self._stdin_qa_loop, daemon=True).start()
        threading.Thread(target=self._qa_worker, daemon=True).start()

        print("正在加载 YOLO 模型...")
        self.detector._load()
        print("YOLO 就绪。问答模式：纯本地知识库，不联网。\n")

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # 视频循环：保留选中状态，用冻结帧或当前帧重置跟踪器
                if self.track.active and self.track.bbox and self.track.frozen_frame is not None:
                    ff = self.track.frozen_frame
                    x1, y1, x2, y2 = self.track.bbox
                    rgb = cv2.cvtColor(ff, cv2.COLOR_BGR2RGB)
                    self.tracker = dlib.correlation_tracker()
                    self.tracker.start_track(rgb, dlib.rectangle(x1, y1, x2, y2))
                continue

            frame = cv2.resize(frame, dim)
            current[0] = frame.copy()

            with self._track_lock:
                if self.track.active:
                    self.track.frozen_frame = frame.copy()
                    self._update_track(frame)
                elif not self._qa_busy:
                    self._all_dets = self.detector.detect(frame)

            vis = self._render(frame, show_all_boxes=show_all)
            if not self.track.active:
                cv2.putText(vis, "Click building | D:all boxes | R:reset | Q:quit",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow(self._window, vis)
            elapsed = int((time.time() - t0) * 1000)
            key = cv2.waitKey(max(1, delay - elapsed)) & 0xFF
            if key == ord("q"):
                self._running = False
                break
            if key == ord("r"):
                with self._track_lock:
                    self.track = TrackState()
                self._qa_building = ""
                print("[重置]")
            if key == ord("d"):
                show_all = not show_all

        cap.release()
        cv2.destroyAllWindows()


def run_video_demo(config_path: str = "config.yaml"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    CockpitVideoDemo(config).run()
