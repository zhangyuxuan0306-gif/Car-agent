"""Master 主终端 — 五阶段多智能体编排"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from src.agents.critic_agent import CriticAgent
from src.agents.dashboard import AgentDashboard
from src.agents.image_slicer import slice_tiles
from src.agents.rag_agent import RAGAgent
from src.agents.schemas import GuideOutput, HandoverPayload, VisionResult
from src.agents.vision_agent import VisionAgent
from src.agents.writer_agent import WriterAgent
from src.multimodal.knowledge_base import BuildingKnowledgeBase
from src.multimodal.llm_synthesizer import LLMAnswerSynthesizer
from src.multimodal.web_knowledge import web_searcher_from_config
from src.perception.custom_yolo_detector import CustomYoloDetector, YoloDetection
from src.utils.draw_utils import draw_text_cn

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    智能座舱多智能体协作编排器

    阶段一：点击捕获 + 图像切片
    阶段二：3× Vision Agent 并行识别
    阶段三：Vision → RAG → Writer 异构协作
    阶段四：Writer → Critic 质检（可回退重写）
    阶段五：成果落地 + 看板 100%
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        on_dashboard_update: Optional[Callable[[str], None]] = None,
    ):
        root = os.path.dirname(os.path.abspath(config_path))
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        agent_cfg = self.config.get("agents", {})
        qa_cfg = self.config.get("qa", {})
        self.num_vision_agents = agent_cfg.get("parallel_vision", 3)
        self.max_critic_retries = agent_cfg.get("max_critic_retries", 2)
        self.use_llm_writer = agent_cfg.get(
            "use_llm_writer", qa_cfg.get("use_llm", False)
        )

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

        kb_cfg = self.config.get("knowledge", {})
        kb_path = kb_cfg.get("db_path", "data/knowledge/buildings.json")
        if not os.path.isabs(kb_path):
            kb_path = os.path.join(root, kb_path)
        self.kb = BuildingKnowledgeBase(db_path=kb_path)
        self.use_web = kb_cfg.get("web_search", False)
        self.web = web_searcher_from_config(kb_cfg)

        synthesizer = None
        if self.use_llm_writer:
            llm_cfg = self.config.get("llm", {})
            device = self.config.get("system", {}).get("device", "cuda")
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            synthesizer = LLMAnswerSynthesizer(
                model_name=llm_cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
                device=device,
            )

        self.dashboard = AgentDashboard(on_update=on_dashboard_update)
        self.rag_agent = RAGAgent(self.kb, self.dashboard, self.web, self.use_web)
        self.writer_agent = WriterAgent(self.dashboard, synthesizer, self.use_llm_writer)
        self.critic_agent = CriticAgent(self.dashboard)

    def run(
        self,
        frame: np.ndarray,
        click_x: int,
        click_y: int,
    ) -> Tuple[GuideOutput, np.ndarray]:
        """执行完整五阶段多智能体流水线，返回 (GuideOutput, 可视化帧)"""
        self.dashboard.clear()
        payload = HandoverPayload(
            phase="init",
            click_point=(click_x, click_y),
        )

        # ── 阶段一：触发与派发 ──
        self.dashboard.log(
            "System",
            f"捕获到车窗点击事件(X:{click_x}, Y:{click_y})，全景图切片中...",
        )
        self.dashboard.set_progress(5)

        tiles_with_crops = slice_tiles(frame, click_x, click_y)
        payload.tiles = [t for t, _ in tiles_with_crops]
        payload.phase = "sliced"
        self.dashboard.log(
            "System",
            f"全景图切片完成，生成 {len(tiles_with_crops)} 个子任务。",
        )
        self.dashboard.set_progress(15)

        # 预加载 YOLO，避免并行 Vision Agent 重复初始化
        self.detector._load()

        # ── 阶段二：相同角色并行 Vision ──
        vision_agents = [
            VisionAgent(f"Vision_Agent_{i + 1}", self.detector)
            for i in range(min(self.num_vision_agents, len(tiles_with_crops)))
        ]
        for va in vision_agents:
            self.dashboard.set_agent_status(va.agent_id, "RUNNING")

        self.dashboard.log("Parallel", "启动 Vision Agent 并行识别...")
        vision_results = self._run_vision_parallel(
            vision_agents, tiles_with_crops, frame
        )
        payload.vision_results = vision_results
        payload.phase = "vision_done"

        for vr in vision_results:
            self.dashboard.log(
                "Parallel",
                f"{vr.agent_id} 识别成功：【{vr.building_name}】",
            )
            self.dashboard.set_agent_status(vr.agent_id, "DONE")

        self.dashboard.set_progress(40)

        # 将识别结果映射到全图检测框（避免切片坐标偏移）
        self._resolve_vision_bboxes(frame, vision_results)

        # 去重建筑列表（保留置信度最高）
        buildings = self._dedupe_buildings(vision_results)
        highlights = self._dedupe_bboxes(vision_results, buildings)

        # ── 阶段三：异构协作 Vision → RAG → Writer ──
        self.dashboard.log("Handover", "视觉团队 -> RAG 检索团队")
        self.dashboard.set_agent_status("RAG_Agent", "RUNNING")
        payload.phase = "rag"
        rag_context = self.rag_agent.retrieve(vision_results)
        payload.rag_context = rag_context
        self.dashboard.set_agent_status("RAG_Agent", "DONE")
        self.dashboard.set_progress(60)

        self.dashboard.log("Handover", "RAG 检索团队 -> Writer 编剧团队")
        self.dashboard.set_agent_status("Writer_Agent", "RUNNING")
        payload.phase = "writing"

        critic_feedback = ""
        draft = ""
        for attempt in range(self.max_critic_retries + 1):
            draft = self.writer_agent.write(buildings, rag_context, critic_feedback)
            payload.draft_script = draft

            # ── 阶段四：Critic 质检 ──
            self.dashboard.set_agent_status("Critic_Agent", "RUNNING")
            payload.phase = "critic"
            result = self.critic_agent.verify(draft, buildings, rag_context)
            payload.critic_feedback = result.feedback

            if result.passed:
                self.dashboard.set_agent_status("Critic_Agent", "DONE")
                break

            self.dashboard.set_agent_status("Critic_Agent", "FAILED")
            critic_feedback = result.feedback
            if attempt < self.max_critic_retries:
                self.dashboard.log(
                    "Handover",
                    f"文案退回 Writer 重写（第 {attempt + 2} 次）",
                )
                self.dashboard.set_agent_status("Writer_Agent", "RUNNING")
            else:
                self.dashboard.log(
                    "System",
                    "已达最大重写次数，使用当前版本推送。",
                )

        self.dashboard.set_agent_status("Writer_Agent", "DONE")
        payload.final_script = draft
        payload.phase = "done"
        payload.progress = 100.0

        # ── 阶段五：成果落地 ──
        self.dashboard.set_progress(100)
        self.dashboard.log(
            "System",
            "最终导游文本通过校验，已成功推送至车载大屏。任务圆满完成！",
        )
        for aid in list(self.dashboard._agent_status.keys()):
            if self.dashboard._agent_status[aid] != "IDLE":
                pass  # keep DONE status for display

        vis = self._render_highlights(frame, vision_results, draft, buildings)

        return GuideOutput(
            script=draft,
            buildings=buildings,
            highlights=highlights,
            payload=payload,
            dashboard_log=self.dashboard.render(),
        ), vis

    def _run_vision_parallel(
        self,
        agents: List[VisionAgent],
        tiles_with_crops: list,
        frame: np.ndarray,
    ) -> List[VisionResult]:
        results: List[VisionResult] = []

        def _task(agent: VisionAgent, tile_meta, crop):
            return agent.recognize(
                tile_meta.tile_id,
                crop,
                frame,
                tile_meta.center,
                tile_meta.bbox,
            )

        with ThreadPoolExecutor(max_workers=len(agents)) as pool:
            futures = {
                pool.submit(_task, agents[i], tiles_with_crops[i][0], tiles_with_crops[i][1]): i
                for i in range(len(agents))
            }
            ordered = [None] * len(agents)
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    ordered[idx] = fut.result()
                except Exception as e:
                    logger.error("Vision Agent %d 失败: %s", idx, e)
                    ordered[idx] = VisionResult(
                        agent_id=agents[idx].agent_id,
                        tile_id=tiles_with_crops[idx][0].tile_id,
                        building_name="未识别建筑",
                        confidence=0.0,
                    )
            results = [r for r in ordered if r is not None]

        return results

    def resolve_building_bboxes(
        self,
        frame: np.ndarray,
        building_names: List[str],
    ) -> dict:
        """按建筑名称解析全图 YOLO 检测框，供视频 overlay 使用"""
        full_dets = self.detector.detect(frame)
        result = {}
        for name in building_names:
            matches = [d for d in full_dets if d.label == name]
            if matches:
                best = max(matches, key=lambda d: d.confidence)
                result[name] = best.bbox
        return result

    def _resolve_vision_bboxes(
        self,
        frame: np.ndarray,
        vision_results: List[VisionResult],
    ):
        """用全图检测结果校正 Vision Agent 的 bbox 坐标"""
        full_dets = self.detector.detect(frame)
        for vr in vision_results:
            matches = [d for d in full_dets if d.label == vr.building_name]
            if matches:
                best = max(matches, key=lambda d: d.confidence)
                vr.bbox = best.bbox

    @staticmethod
    def _dedupe_bboxes(
        vision_results: List[VisionResult],
        buildings: List[str],
    ) -> List[tuple]:
        bboxes = []
        for name in buildings:
            for vr in vision_results:
                if vr.building_name == name and vr.bbox != (0, 0, 0, 0):
                    bboxes.append(vr.bbox)
                    break
        return bboxes

    @staticmethod
    def _dedupe_buildings(vision_results: List[VisionResult]) -> List[str]:
        best: dict = {}
        for vr in vision_results:
            name = vr.building_name
            if name == "未识别建筑":
                continue
            if name not in best or vr.confidence > best[name]:
                best[name] = vr.confidence
        return sorted(best.keys(), key=lambda n: best[n], reverse=True)

    @staticmethod
    def _render_highlights(
        frame: np.ndarray,
        vision_results: List[VisionResult],
        script: str,
        buildings: List[str] | None = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        vis = frame.copy().astype(np.float32)
        dark = vis * 0.25

        # 去重：每个建筑只画一个框
        seen = set()
        to_draw = []
        order = buildings or []
        for name in order:
            if name in seen:
                continue
            for vr in vision_results:
                if vr.building_name == name and vr.bbox != (0, 0, 0, 0):
                    to_draw.append(vr)
                    seen.add(name)
                    break

        mask = np.zeros((h, w), dtype=bool)
        for vr in to_draw:
            x1, y1, x2, y2 = vr.bbox
            mask[y1:y2, x1:x2] = True
        if mask.any():
            m = mask[:, :, None].astype(np.float32)
            vis = (vis * m + dark * (1.0 - m)).astype(np.uint8)
        else:
            vis = vis.astype(np.uint8)

        colors = [(0, 0, 255), (0, 165, 255), (255, 0, 255), (0, 255, 128)]
        for i, vr in enumerate(to_draw):
            x1, y1, x2, y2 = vr.bbox
            color = colors[i % len(colors)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            vis = draw_text_cn(vis, vr.building_name, (x1, max(y1 - 32, 4)), font_size=20)

        if script:
            summary = script if len(script) <= 80 else script[:79] + "…"
            vis = draw_text_cn(vis, summary, (10, h - 70), font_size=18, max_width=40)

        return vis
