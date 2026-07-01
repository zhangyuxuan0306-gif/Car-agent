"""RAG Agent — 知识检索智能体（本地知识库 + 可选联网补全）"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.agents.dashboard import AgentDashboard
from src.agents.schemas import VisionResult
from src.multimodal.knowledge_base import BuildingKnowledgeBase
from src.multimodal.web_knowledge import WebKnowledgeSearcher

logger = logging.getLogger(__name__)


class RAGAgent:
    """根据 Vision 识别结果检索本地知识库，可选联网补全"""

    def __init__(
        self,
        kb: BuildingKnowledgeBase,
        dashboard: AgentDashboard,
        web: Optional[WebKnowledgeSearcher] = None,
        use_web: bool = False,
    ):
        self.kb = kb
        self.dashboard = dashboard
        self.web = web if use_web else None
        self.use_web = use_web

    def retrieve(self, vision_results: List[VisionResult]) -> Dict[str, str]:
        """为每个识别到的建筑检索深度背景资料"""
        context: Dict[str, str] = {}
        seen = set()

        for vr in vision_results:
            name = vr.building_name
            if not name or name == "未识别建筑" or name in seen:
                continue
            seen.add(name)

            self.dashboard.log(
                "Handover",
                f"RAG_Agent 正在调取【{name}】的知识库数据...",
            )
            ctx = self.kb.get_context_for_building(name)
            if not ctx:
                hits = self.kb.search(name, top_k=1)
                ctx = hits[0].to_context() if hits else ""

            if self.use_web and self.web and len(ctx) < 120:
                self.dashboard.log("Handover", f"RAG_Agent 联网补全【{name}】...")
                online = self.web.search(name)
                if online:
                    online_block = f"【联网资料】\n{online}"
                    ctx = f"{ctx}\n\n{online_block}" if ctx else f"建筑名称: {name}\n{online_block}"

            if not ctx:
                ctx = f"建筑名称: {name}\n简介: 北京CBD区域地标建筑。"

            context[name] = ctx

        return context
