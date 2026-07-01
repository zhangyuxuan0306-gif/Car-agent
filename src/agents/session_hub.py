"""多 Agent 会话同步中心 — 静默感知缓存 + 全局聊天记录"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.perception.custom_yolo_detector import YoloDetection


@dataclass
class ChatMessage:
    building: str
    role: str       # user / assistant / system
    content: str
    source: str = ""  # local / web / llm
    timestamp: float = field(default_factory=time.time)


@dataclass
class BuildingAgentRecord:
    """单个 Vision Agent 绑定的建筑状态（默认不可见）"""
    agent_id: str
    building_name: str
    detection: Optional[YoloDetection] = None
    intro: str = ""
    rag_context: str = ""
    visible: bool = False
    activated: bool = False
    chat: List[ChatMessage] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)


class AgentSessionHub:
    """
    全局会话 Hub：
    - 后台多个 Vision Agent 预识别结果写入此处，UI 不展示
    - 用户点击后激活对应 Agent 并显示介绍
    - 所有问答写入全局历史，并同步到各 Agent
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._agents: Dict[str, BuildingAgentRecord] = {}
        self._global_chat: List[ChatMessage] = []
        self._active_building: str = ""

    def upsert_perception(
        self,
        agent_id: str,
        detection: YoloDetection,
        intro: str,
        rag_context: str,
    ) -> None:
        """后台静默写入/更新（不改变 visible 状态）"""
        with self._lock:
            name = detection.label
            rec = self._agents.get(name)
            if rec is None:
                rec = BuildingAgentRecord(
                    agent_id=agent_id,
                    building_name=name,
                    detection=detection,
                    intro=intro,
                    rag_context=rag_context,
                    visible=False,
                )
                self._agents[name] = rec
            else:
                rec.detection = detection
                rec.intro = intro or rec.intro
                rec.rag_context = rag_context or rec.rag_context
                rec.last_update = time.time()

    def activate_at_point(
        self,
        px: int,
        py: int,
        fallback_det: Optional[YoloDetection] = None,
    ) -> Optional[BuildingAgentRecord]:
        """点击激活：命中缓存建筑则 0 延迟唤醒对应 Agent"""
        with self._lock:
            # 优先：点选检测框内
            hit = None
            for rec in self._agents.values():
                if rec.detection and rec.detection.contains(px, py):
                    if hit is None or rec.detection.confidence > hit.detection.confidence:
                        hit = rec

            if hit is None and fallback_det is not None:
                hit = self._agents.get(fallback_det.label)
                if hit is None:
                    hit = BuildingAgentRecord(
                        agent_id=f"Vision_Agent_{len(self._agents) + 1}",
                        building_name=fallback_det.label,
                        detection=fallback_det,
                    )
                    self._agents[fallback_det.label] = hit
                else:
                    hit.detection = fallback_det

            if hit is None:
                return None

            for rec in self._agents.values():
                rec.visible = False
            hit.visible = True
            hit.activated = True
            self._active_building = hit.building_name
            return hit

    def add_message(
        self,
        building: str,
        role: str,
        content: str,
        source: str = "",
        broadcast: bool = True,
    ) -> None:
        """记录聊天并同步到全局与其他 Agent"""
        msg = ChatMessage(building=building, role=role, content=content, source=source)
        with self._lock:
            self._global_chat.append(msg)
            if building in self._agents:
                self._agents[building].chat.append(msg)
            if broadcast:
                sync_note = ChatMessage(
                    building=building,
                    role="system",
                    content=f"[同步] {building}：{content[:80]}{'…' if len(content) > 80 else ''}",
                    source="sync",
                )
                for name, rec in self._agents.items():
                    if name != building:
                        rec.chat.append(sync_note)

    def get_active(self) -> Optional[BuildingAgentRecord]:
        with self._lock:
            if not self._active_building:
                return None
            return self._agents.get(self._active_building)

    def get_chat_history(self, max_turns: int = 10) -> List[Tuple[str, str]]:
        """供 LLM 多轮对话使用的共享记忆（跨 Agent）"""
        with self._lock:
            pairs: List[Tuple[str, str]] = []
            for m in self._global_chat:
                if m.role in ("user", "assistant") and m.source != "cache":
                    pairs.append((m.role, m.content))
            return pairs[-max_turns * 2:]

    def get_shared_context(self, building: str, max_turns: int = 8) -> str:
        """供 RAG/LLM 使用的跨 Agent 共享上下文"""
        with self._lock:
            lines = ["【跨Agent共享记忆】"]
            for m in self._global_chat[-max_turns * 2:]:
                if m.role == "system":
                    continue
                lines.append(f"[{m.building}·{m.role}] {m.content}")
            if building in self._agents:
                rag = self._agents[building].rag_context
                if rag:
                    lines.append(f"【本地RAG·{building}】\n{rag[:600]}")
            return "\n".join(lines)

    def snapshot_silent(self) -> List[str]:
        """后台状态摘要（终端看板用，不画到视频）"""
        with self._lock:
            return [
                f"{r.agent_id}:{r.building_name}{'*' if r.visible else ''}"
                for r in sorted(self._agents.values(), key=lambda x: x.agent_id)
            ]

    def reset(self) -> None:
        with self._lock:
            self._agents.clear()
            self._global_chat.clear()
            self._active_building = ""
