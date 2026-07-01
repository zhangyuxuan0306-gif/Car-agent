"""多智能体交互视频 — 后台静默预识别 + 点击激活 + 会话同步"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.agents.session_hub import AgentSessionHub
from src.agents.silent_perception import SilentPerceptionWorker
from src.utils.draw_utils import draw_progress_bar
from src.video.cockpit_video import CockpitVideoDemo, TrackState

logger = logging.getLogger(__name__)


class CockpitAgentVideoDemo(CockpitVideoDemo):
    """
    - 启动：后台异步预加载模型 + 多 Vision Agent 静默识别（不显示）
    - 点击：瞬时命中缓存，激活对应 Agent 显示介绍
    - 问答：本地 RAG → 联网，聊天记录全局同步
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._window = "Smart Cockpit - Multi-Agent"
        self._hub = AgentSessionHub()
        self._silent: SilentPerceptionWorker | None = None
        self._models_ready = False
        self._preload_status = "初始化中…"

    def _on_click(self, frame, x: int, y: int):
        h, w = frame.shape[:2]
        x, y = int(max(0, min(x, w - 1))), int(max(0, min(y, h - 1)))

        dets = self._all_dets or self.detector.detect(frame)
        fallback = self.detector.pick_at_point(frame, x, y, dets=dets)

        rec = self._hub.activate_at_point(x, y, fallback_det=fallback)
        if rec is None or rec.detection is None:
            print("[提示] 未检测到建筑，请对准建筑物点击", flush=True)
            return

        det = rec.detection
        intro = rec.intro or self.brief.describe(det.label)

        # 立即跟踪 + 显示缓存介绍（0 延迟视觉）
        self._start_track(frame, det, (x, y))
        with self._track_lock:
            self.track.description = intro
            self.track.qa_answer = ""

        self._qa_building = det.label
        self._hub.add_message(
            det.label, "assistant", intro, source="cache", broadcast=True
        )

        silent = self._hub.snapshot_silent()
        print(f"\n[点击] ({x},{y}) → 激活 {rec.agent_id} 【{det.label}】", flush=True)
        print(f"[缓存命中] 介绍：{intro}", flush=True)
        print(f"[后台Agent] {' | '.join(silent)}", flush=True)
        print("[提示] 终端输入问题，像聊大模型一样对话（本地RAG→联网→LLM）\n", flush=True)

    def _qa_worker(self):
        """Agent 对话：本地RAG + 联网 + LLM，共享记忆同步"""
        while self._running:
            try:
                item = self._qa_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break

            building, question = item
            self._qa_busy = True
            self._set_qa_progress(0.05, "准备对话")

            self._hub.add_message(building, "user", question, source="user", broadcast=False)

            def on_progress(p: float, msg: str):
                self._set_qa_progress(p, msg)

            t0 = time.time()
            try:
                history = self._hub.get_chat_history()
                memory = self._hub.get_shared_context(building)
                ans, src = self.brief.answer_agent_chat(
                    building,
                    question,
                    chat_history=history,
                    shared_memory=memory,
                    on_progress=on_progress,
                )
                elapsed = time.time() - t0
                self._hub.add_message(building, "assistant", ans, source=src, broadcast=True)

                print(f"\n[Agent·{building} · {src} · {elapsed:.1f}s]\n{ans}", flush=True)
                mem_count = len(self._hub.get_chat_history())
                print(f"[共享记忆] 全局 {mem_count} 轮 · 已同步至所有后台 Agent", flush=True)
                print(flush=True)

                with self._track_lock:
                    if self.track.active and self.track.name == building:
                        self.track.qa_answer = ans
            except Exception as e:
                logger.error("问答失败: %s", e, exc_info=True)
                err = f"抱歉，处理问题时出错：{e}"
                print(f"[问答] {err}", flush=True)
                with self._track_lock:
                    if self.track.active and self.track.name == building:
                        self.track.qa_answer = err
            finally:
                time.sleep(0.25)
                self._qa_busy = False
                self._set_qa_progress(0.0, "")
                self._qa_queue.task_done()

    def _refine_with_yolo(self, frame):
        if self._qa_busy:
            return
        super()._refine_with_yolo(frame)

    def _render(self, frame, show_all_boxes: bool = False):
        # 静默模式：不显示所有检测框（show_all 除外）
        vis = super()._render(frame, show_all_boxes=False)
        if self._qa_busy and self.track.active:
            vis = draw_progress_bar(vis, self._qa_progress, self._qa_status)
        elif not self._models_ready and not self.track.active:
            cv2.putText(
                vis, f"{self._preload_status}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
        return vis

    def run(self):
        path = self._resolve_video()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开: {path}")

        dim = (self.width, self.height)
        delay = max(1, int(1000 / self.fps))
        current = [None]

        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window, 960, 540)

        def mouse_cb(event, mx, my, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and current[0] is not None:
                self._on_click(current[0], mx, my)

        cv2.setMouseCallback(self._window, mouse_cb)

        print("\n=== 智能座舱 · 静默感知多 Agent ===")
        print("后台预识别 | 点击激活 | 本地RAG→联网 | 会话同步\n")

        self._running = True
        threading.Thread(target=self._stdin_qa_loop, daemon=True).start()
        threading.Thread(target=self._qa_worker, daemon=True).start()

        # 异步预加载
        def _preload():
            self._preload_status = "YOLO 加载中…"
            self.detector._load()
            self._preload_status = "LLM 预加载中…"
            if self.brief.use_llm:
                try:
                    self.brief.preload_llm_async()
                    # 等待 LLM 就绪，避免首次问答无响应
                    if self.brief.synthesizer:
                        self.brief.synthesizer.load()
                except Exception as e:
                    logger.warning("LLM 预加载: %s", e)
            self._models_ready = True
            self._preload_status = "就绪"
            mode = "LLM对话+联网" if self.brief.use_llm and self.brief.use_web else (
                "LLM对话" if self.brief.use_llm else "本地RAG"
            )
            print(f"[System] 模型就绪（{mode}），后台静默感知运行中", flush=True)

        threading.Thread(target=_preload, daemon=True).start()

        self._silent = SilentPerceptionWorker(
            detector=self.detector,
            brief=self.brief,
            kb=self.brief.kb,
            hub=self._hub,
            interval=0.55,
        )
        self._silent.start()

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if self.track.active and self.track.bbox and self.track.frozen_frame is not None:
                    ff = self.track.frozen_frame
                    x1, y1, x2, y2 = self.track.bbox
                    import dlib
                    rgb = cv2.cvtColor(ff, cv2.COLOR_BGR2RGB)
                    self.tracker = dlib.correlation_tracker()
                    self.tracker.start_track(rgb, dlib.rectangle(x1, y1, x2, y2))
                continue

            frame = cv2.resize(frame, dim)
            current[0] = frame.copy()

            # 喂帧给后台静默感知（不画框）
            if self._silent and self._models_ready:
                self._silent.submit_frame(frame)

            with self._track_lock:
                if self.track.active:
                    self.track.frozen_frame = frame.copy()
                    self._update_track(frame)
                elif self._models_ready and not self._qa_busy:
                    self._all_dets = self.detector.detect(frame)

            vis = self._render(frame)
            if not self.track.active:
                hint = "Click building | R:reset | Q:quit"
                if self._models_ready:
                    snap = self._hub.snapshot_silent()
                    if snap:
                        hint += f"  [静默:{len(snap)}]"
                cv2.putText(
                    vis, hint, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )

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
                self._hub.reset()
                print("[重置]", flush=True)

        if self._silent:
            self._silent.stop()
        cap.release()
        cv2.destroyAllWindows()


def run_agent_video_demo(config: dict):
    CockpitAgentVideoDemo(config).run()
