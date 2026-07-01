"""后台静默感知 — 多 Vision Agent 并发预识别，不渲染到画面"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.agents.session_hub import AgentSessionHub
from src.multimodal.building_brief import BuildingBriefService
from src.multimodal.knowledge_base import BuildingKnowledgeBase
from src.perception.custom_yolo_detector import CustomYoloDetector

logger = logging.getLogger(__name__)


class SilentPerceptionWorker:
    """后台线程：周期性 YOLO 检测 + 并行写入 SessionHub"""

    def __init__(
        self,
        detector: CustomYoloDetector,
        brief: BuildingBriefService,
        kb: BuildingKnowledgeBase,
        hub: AgentSessionHub,
        interval: float = 0.6,
        max_agents: int = 4,
    ):
        self.detector = detector
        self.brief = brief
        self.kb = kb
        self.hub = hub
        self.interval = interval
        self.max_agents = max_agents
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._pool = ThreadPoolExecutor(max_workers=max_agents)

    def submit_frame(self, frame) -> None:
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="silent-perception")
        self._thread.start()
        logger.info("后台静默感知已启动")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._pool.shutdown(wait=False)

    def _enrich(self, agent_id: str, det) -> None:
        intro = self.brief.describe(det.label)
        rag = self.kb.get_context_for_building(det.label)
        self.hub.upsert_perception(agent_id, det, intro, rag)

    def _loop(self) -> None:
        while self._running:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                dets = self.detector.detect(frame)
                dets = dets[: self.max_agents]
                futures = []
                for i, det in enumerate(dets):
                    aid = f"Vision_Agent_{i + 1}"
                    futures.append(self._pool.submit(self._enrich, aid, det))
                for f in futures:
                    try:
                        f.result(timeout=8)
                    except Exception as e:
                        logger.debug("静默感知任务失败: %s", e)
            except Exception as e:
                logger.debug("静默感知帧失败: %s", e)
            time.sleep(self.interval)
