"""交互问答引擎 — 本地+联网资料，大模型整理回答"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.multimodal.building_brief import BuildingBriefService

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class QAEngine:
    """交互式问答引擎"""

    brief: BuildingBriefService
    max_history: int = 10
    _history: List[ChatMessage] = field(default_factory=list, repr=False)
    _building: str = field(default="", repr=False)

    def set_context(
        self,
        image=None,
        building_label: str = "",
        description: str = "",
        knowledge_context: str = "",
    ):
        """设置当前关注建筑（image 保留兼容，问答不依赖 VLM）"""
        self._building = building_label or ""

    def ask(self, question: str) -> str:
        if not question.strip():
            return "请输入问题"
        if not self._building:
            return "请先选择/识别一栋建筑"

        answer = self.brief.answer(self._building, question)
        self._history.append(ChatMessage(role="user", content=question))
        self._history.append(ChatMessage(role="assistant", content=answer))
        if len(self._history) > self.max_history * 2:
            self._history = self._history[-self.max_history * 2 :]
        return answer

    def clear_history(self):
        self._history.clear()

    def get_history_display(self) -> List[Tuple[str, str]]:
        pairs = []
        for i in range(0, len(self._history) - 1, 2):
            if i + 1 < len(self._history):
                pairs.append((self._history[i].content, self._history[i + 1].content))
        return pairs
