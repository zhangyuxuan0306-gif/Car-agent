"""Agent 间结构化交接协议（JSON Schema 对应 dataclass）"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class ImageTile:
    """阶段一切片结果"""
    tile_id: str          # A / B / C
    bbox: tuple            # x1, y1, x2, y2
    center: tuple          # cx, cy

    def to_dict(self) -> dict:
        return {"tile_id": self.tile_id, "bbox": list(self.bbox), "center": list(self.center)}


@dataclass
class VisionResult:
    """Vision Agent 识别结果"""
    agent_id: str
    tile_id: str
    building_name: str
    confidence: float
    bbox: tuple = (0, 0, 0, 0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HandoverPayload:
    """阶段间交接 JSON 状态文件"""
    phase: str
    click_point: tuple
    tiles: List[ImageTile] = field(default_factory=list)
    vision_results: List[VisionResult] = field(default_factory=list)
    rag_context: Dict[str, str] = field(default_factory=dict)
    draft_script: str = ""
    critic_feedback: str = ""
    final_script: str = ""
    progress: float = 0.0

    def to_json(self, indent: int = 2) -> str:
        d = {
            "phase": self.phase,
            "click_point": list(self.click_point),
            "tiles": [t.to_dict() for t in self.tiles],
            "vision_results": [v.to_dict() for v in self.vision_results],
            "rag_context": self.rag_context,
            "draft_script": self.draft_script,
            "critic_feedback": self.critic_feedback,
            "final_script": self.final_script,
            "progress": self.progress,
        }
        return json.dumps(d, ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> "HandoverPayload":
        d = json.loads(raw)
        return cls(
            phase=d.get("phase", ""),
            click_point=tuple(d.get("click_point", (0, 0))),
            tiles=[
                ImageTile(t["tile_id"], tuple(t["bbox"]), tuple(t["center"]))
                for t in d.get("tiles", [])
            ],
            vision_results=[
                VisionResult(**v) for v in d.get("vision_results", [])
            ],
            rag_context=d.get("rag_context", {}),
            draft_script=d.get("draft_script", ""),
            critic_feedback=d.get("critic_feedback", ""),
            final_script=d.get("final_script", ""),
            progress=d.get("progress", 0.0),
        )


@dataclass
class CriticResult:
    passed: bool
    feedback: str = ""
    issues: List[str] = field(default_factory=list)


@dataclass
class GuideOutput:
    """阶段五最终成果"""
    script: str
    buildings: List[str]
    highlights: List[tuple]   # bbox list for HMI overlay
    payload: HandoverPayload
    dashboard_log: str = ""
