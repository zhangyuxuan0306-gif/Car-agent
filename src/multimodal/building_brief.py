"""建筑简介/问答 — 点击简介≤30字；问答默认纯本地快速回答"""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Dict, List, Optional, Tuple

from src.data.building_intros import BUILDING_INTROS
from src.multimodal.knowledge_base import BuildingKnowledgeBase
from src.multimodal.local_qa_engine import LocalQAEngine, ProgressFn
from src.multimodal.llm_synthesizer import LLMAnswerSynthesizer
from src.multimodal.question_router import (
    answer_from_memory,
    classify_question,
    poi_fallback_answer,
    poi_keywords,
    poi_topic_label,
)
from src.multimodal.web_knowledge import WebKnowledgeSearcher

logger = logging.getLogger(__name__)

MAX_BRIEF_LEN = 30


def _answer_matches_question(ans: str, question: str) -> bool:
    """本地回答是否与问题相关；不相关则应联网或大模型"""
    if any(k in question for k in ("多高", "多少米", "几米", "高度")):
        return "米" in ans or "高" in ans
    if any(k in question for k in ("谁设计", "设计师", "建筑师", "谁建")):
        return "设计" in ans or "建筑师" in ans
    if any(k in question for k in ("什么时候", "哪年", "建成", "竣工")):
        return "年" in ans
    if any(k in question for k in ("在哪", "哪里", "位置", "地址")):
        return "位于" in ans or "CBD" in ans or "北京" in ans

    topical = (
        "外墙", "玻璃", "幕墙", "外立面", "立面", "材料",
        "干什么", "用途", "功能", "做什么",
        "风格", "造型", "外观", "特点", "特色",
        "是什么", "介绍", "简介", "历史", "文化",
    )
    q_kws = [k for k in topical if k in question]
    if not q_kws:
        q_kws = [w for w in re.split(r"[，。；？?！!、\s]+", question) if len(w) >= 2]
    if not q_kws:
        return True
    hits = sum(1 for k in q_kws if k in ans)
    return hits >= max(1, len(q_kws) // 2)


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

    def _building_location(self, building: str) -> str:
        hits = self.kb.search(building, top_k=1)
        if not hits:
            return "北京CBD"
        meta = hits[0].metadata
        loc = meta.get("location") or hits[0].city or "北京CBD"
        return str(loc)

    def answer_rag_first(
        self,
        building: str,
        question: str,
        shared_context: str = "",
        on_progress: ProgressFn = None,
    ) -> Tuple[str, str]:
        """
        智能座舱 Agent 问答：本地 RAG 优先，不足则联网。
        返回 (回答文本, 来源 local/web/none)
        """
        if not question.strip():
            return "请输入问题", "none"
        q = question.strip()
        if on_progress:
            on_progress(0.15, "检索本地RAG")

        ans = self.local_qa.answer(building, q, on_progress=None)
        local_ok = bool(ans and "暂无" not in ans and _answer_matches_question(ans, q))

        if local_ok:
            if on_progress:
                on_progress(1.0, "完成(本地)")
            return ans, "local"

        if self.use_web and self.web:
            if on_progress:
                on_progress(0.45, "联网检索中")
            query = f"{building} {q}"
            if shared_context:
                query = f"{building} {q}"
            online = self.web.search_answer(building, q)
            if online:
                body = online if building in online else f"{building}：{online}"
                if on_progress:
                    on_progress(1.0, "完成(联网)")
                return body[:450] + ("…" if len(body) > 450 else ""), "web"

        if on_progress:
            on_progress(1.0, "完成")
        if not local_ok:
            return f"暂未找到关于「{building}」与「{q}」的相关资料。", "none"
        return ans or f"暂未找到关于「{building}」的资料。", "none"

    def answer_agent_chat(
        self,
        building: str,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        shared_memory: str = "",
        on_progress: ProgressFn = None,
    ) -> Tuple[str, str]:
        """
        Agent 对话模式：本地 RAG → 联网补资料 → 大模型多轮聊天。
        返回 (回答, 来源标签)
        """
        if not question.strip():
            return "请输入问题", "none"
        q = question.strip()
        chat_history = chat_history or []

        intent = classify_question(q)
        if intent == "memory":
            if on_progress:
                on_progress(0.25, "读取对话记忆")
            mem_ans = answer_from_memory(q, chat_history, building)
            if on_progress:
                on_progress(1.0, "完成(记忆)")
            return mem_ans or "暂时没有找到相关的对话记录。", "memory"

        if intent == "chitchat":
            if on_progress:
                on_progress(0.3, "对话中")
            if self.use_llm:
                if self.synthesizer is None:
                    self.synthesizer = LLMAnswerSynthesizer()
                try:
                    self.synthesizer.load()
                    ans = self.synthesizer.chat(building, q, "", chat_history)
                    if ans:
                        if on_progress:
                            on_progress(1.0, "完成")
                        return ans, "llm"
                except Exception as e:
                    logger.warning("寒暄 LLM 失败: %s", e)
            if on_progress:
                on_progress(1.0, "完成")
            if "谢谢" in q or "感谢" in q:
                return "不客气，有需要随时问我。", "chitchat"
            if "你好" in q or "您好" in q:
                return f"您好！我是智能座舱建筑导游，正在为您介绍{'「' + building + '」' if building else '窗外建筑'}，请问想了解什么？", "chitchat"
            return "您好，我可以帮您介绍窗外建筑的历史、设计和特点。", "chitchat"

        if intent == "nearby":
            location = self._building_location(building)
            topic = poi_topic_label(q)
            web_hit = ""
            if self.use_web and self.web:
                if on_progress:
                    on_progress(0.35, "联网检索周边")
                web_hit = self.web.search_poi(
                    building, q, location=location, keywords=poi_keywords(q)
                )
            if web_hit:
                if self.use_llm:
                    if on_progress:
                        on_progress(0.72, "整理周边信息")
                    if self.synthesizer is None:
                        self.synthesizer = LLMAnswerSynthesizer()
                    try:
                        self.synthesizer.load()
                        ctx = (
                            f"【联网资料·{building}周边{topic}】\n{web_hit}\n\n"
                            f"请仅根据联网资料列出具体店名/地点，不要编造；"
                            f"若资料未提及具体名称，请如实说明。"
                        )
                        ans = self.synthesizer.chat(building, q, ctx, chat_history)
                        if ans and "暂未" not in ans[:8]:
                            if on_progress:
                                on_progress(1.0, "完成(联网)")
                            return ans, "llm+web"
                    except Exception as e:
                        logger.warning("POI LLM 整理失败: %s", e)
                if on_progress:
                    on_progress(1.0, "完成(联网)")
                return f"「{building}」周边{topic}参考：\n{web_hit[:480]}", "web"
            if on_progress:
                on_progress(1.0, "完成")
            return poi_fallback_answer(building, q), "none"

        parts: List[str] = []

        if shared_memory:
            parts.append(shared_memory)

        if on_progress:
            on_progress(0.12, "检索本地RAG")
        local_ans = self.local_qa.answer(building, q, on_progress=None)
        local_ok = bool(local_ans and "暂无" not in local_ans and _answer_matches_question(local_ans, q))
        if local_ok:
            parts.append(f"【本地RAG】\n{local_ans}")
        for hit in self.kb.search(f"{building} {q}", top_k=2):
            parts.append(f"【知识库】\n{hit.to_context()}")

        web_hit = ""
        if self.use_web and self.web:
            if on_progress:
                on_progress(0.32, "联网检索中")
            web_hit = self.web.search_answer(building, q) or ""

        if web_hit:
            parts.append(f"【联网资料】\n{web_hit}")
        elif local_ok:
            pass
        elif local_ans and "暂无" not in local_ans:
            parts.append(f"【本地RAG】\n{local_ans}")

        context = "\n\n".join(parts)

        # 大模型对话（核心）
        if self.use_llm:
            if on_progress:
                on_progress(0.55, "大模型对话中")
            if self.synthesizer is None:
                self.synthesizer = LLMAnswerSynthesizer()
            try:
                self.synthesizer.load()
            except Exception as e:
                logger.warning("LLM 加载失败: %s", e)
            ans = self.synthesizer.chat(building, q, context, chat_history)
            if ans and "暂未找到" not in ans[:8]:
                src = "llm"
                if web_hit:
                    src = "llm+web"
                elif local_ans:
                    src = "llm+local"
                if on_progress:
                    on_progress(1.0, "完成")
                return ans, src

        # 无 LLM 时降级
        if web_hit:
            body = web_hit if building in web_hit else f"{building}：{web_hit}"
            if on_progress:
                on_progress(1.0, "完成(联网)")
            return body[:450], "web"
        if local_ok or (local_ans and _answer_matches_question(local_ans, q)):
            if on_progress:
                on_progress(1.0, "完成(本地)")
            return local_ans, "local"
        if on_progress:
            on_progress(1.0, "完成")
        return f"关于「{building}」的「{q}」，暂未检索到足够资料，请换个问法试试。", "none"

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

        # 纯本地快速路径；启用联网时在本地不足时自动 fallback
        if self.offline_qa and not self.use_llm:
            if on_progress:
                on_progress(0.15, "检索本地知识库")
            ans = self.local_qa.answer(building, q, on_progress=None)

            if self.use_web and self.web:
                web_keywords = (
                    "历史", "故事", "典故", "背景", "为什么", "文化",
                    "价值", "著名", "介绍", "讲讲", "详细", "最新",
                )
                need_web = (
                    not ans
                    or "暂无" in ans
                    or len(ans) < 12
                    or not _answer_matches_question(ans, q)
                    or any(k in q for k in web_keywords)
                )
                if need_web:
                    if on_progress:
                        on_progress(0.45, "联网检索中")
                    online = self.web.search_answer(building, q)
                    if online:
                        body = online if online.startswith(building) else f"{building}：{online}"
                        ans = body[:400] + ("…" if len(body) > 400 else "")
                        if ans and not ans.endswith("。"):
                            ans += "。"

            if on_progress:
                on_progress(1.0, "完成")
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
            online = self.web.search_answer(building, question)
            if online:
                parts.append(f"【联网】{online}")
        return "\n\n".join(parts)

    def preload_llm(self):
        if self.synthesizer:
            self.synthesizer.load()

    def preload_llm_async(self):
        if self.synthesizer:
            self.synthesizer.preload_async()
