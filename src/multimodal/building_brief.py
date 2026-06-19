"""建筑简介/问答 — 点击简介≤30字；问答默认纯本地快速回答"""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Dict, List, Optional

from src.data.building_intros import BUILDING_INTROS
from src.multimodal.knowledge_base import BuildingKnowledgeBase
from src.multimodal.local_qa_engine import LocalQAEngine, ProgressFn
from src.multimodal.llm_synthesizer import LLMAnswerSynthesizer
from src.multimodal.web_knowledge import WebKnowledgeSearcher

logger = logging.getLogger(__name__)

MAX_BRIEF_LEN = 30


def truncate_brief(text: str, max_len: int = MAX_BRIEF_LEN) -> str:
    text = re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", (text or "").strip())
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


class BuildingBriefService:
    """点击简介≤30字；问答：默认纯本地（毫秒级），可选大模型/联网"""

    def __init__(
        self,
        kb: Optional[BuildingKnowledgeBase] = None,
        web: Optional[WebKnowledgeSearcher] = None,
        synthesizer: Optional[LLMAnswerSynthesizer] = None,
        max_brief_len: int = MAX_BRIEF_LEN,
        offline_qa: bool = True,
        use_llm: bool = False,
        use_web: bool = False,
    ):
        self.kb = kb or BuildingKnowledgeBase()
        self.web = web if use_web else None
        self.synthesizer = synthesizer if use_llm else None
        self.max_brief_len = max_brief_len
        self.offline_qa = offline_qa
        self.use_llm = use_llm
        self.use_web = use_web
        self.local_qa = LocalQAEngine(kb=self.kb)
        self._answer_cache: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _local_text(self, building: str) -> str:
        if building in BUILDING_INTROS:
            return BUILDING_INTROS[building]
        ctx = self.kb.get_context_for_building(building)
        if ctx:
            if "简介:" in ctx:
                return ctx.split("简介:")[-1].strip()
            return ctx.replace("\n", " ")
        return ""

    def describe(self, building: str) -> str:
        local = self._local_text(building)
        if local:
            return truncate_brief(local, self.max_brief_len)
        if self.web:
            online = self.web.search(building)
            if online:
                return truncate_brief(online, self.max_brief_len)
        return truncate_brief(f"{building}，北京CBD地标建筑。", self.max_brief_len)

    def answer(
        self,
        building: str,
        question: str,
        on_progress: ProgressFn = None,
    ) -> str:
        if not question.strip():
            return "请输入问题"
        building = building or ""
        q = question.strip()
        cache_key = f"{building}|{q}"

        with self._lock:
            if cache_key in self._answer_cache:
                if on_progress:
                    on_progress(0.5, "读取缓存")
                    on_progress(1.0, "完成")
                return self._answer_cache[cache_key]

        # 纯本地快速路径（默认）
        if self.offline_qa and not self.use_llm:
            ans = self.local_qa.answer(building, q, on_progress=on_progress)
            with self._lock:
                self._answer_cache[cache_key] = ans
            return ans

        # 可选：联网 + 大模型（慢）
        if on_progress:
            on_progress(0.2, "汇总资料")
        context = self._gather_context(building, q)
        if on_progress:
            on_progress(0.5, "大模型整理")
        if self.synthesizer is None:
            self.synthesizer = LLMAnswerSynthesizer()
        ans = self.synthesizer.synthesize(building, q, context)
        if on_progress:
            on_progress(1.0, "完成")
        with self._lock:
            self._answer_cache[cache_key] = ans
        return ans

    def _gather_context(self, building: str, question: str) -> str:
        parts: List[str] = []
        local = self._local_text(building)
        if local:
            parts.append(f"【本地】{local}")
        for hit in self.kb.search(f"{building} {question}", top_k=2):
            parts.append(f"【知识库】{hit.to_context()}")
        if self.web:
            online = self.web.search_answer(building, question) or self.web.search(building)
            if online:
                parts.append(f"【联网】{online}")
        return "\n\n".join(parts)

    def preload_llm(self):
        if self.synthesizer:
            self.synthesizer.load()

    def preload_llm_async(self):
        if self.synthesizer:
            self.synthesizer.preload_async()
