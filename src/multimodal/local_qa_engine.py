"""纯本地快速问答 — 不联网、不加载大模型"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

from src.multimodal.knowledge_base import BuildingKnowledgeBase, KnowledgeEntry

logger = logging.getLogger(__name__)

ProgressFn = Optional[Callable[[float, str], None]]

# (意图关键词, 字段优先级)
_INTENTS: List[Tuple[tuple, tuple]] = [
    (("多高", "多少米", "高度", "几米", "高吗"), ("height_m", "description")),
    (("谁设计", "设计师", "建筑师", "谁建", "设计者"), ("designer", "architect", "description")),
    (("什么时候", "哪年", "建成", "竣工", "建造"), ("built_year", "description")),
    (("在哪", "哪里", "位置", "地址"), ("location", "city", "description")),
    (("干什么", "用途", "功能", "做什么", "干嘛"), ("function", "type", "description")),
    (("风格", "造型", "外观", "特点", "特色"), ("style", "description")),
    (("外墙", "玻璃", "幕墙", "外立面", "立面", "材料", "表面"), ("style", "description")),
    (("是什么", "介绍", "简介", "讲讲"), ("description", "type", "function")),
]


class LocalQAEngine:
    """基于结构化知识库的毫秒级本地问答"""

    def __init__(self, kb: Optional[BuildingKnowledgeBase] = None):
        self.kb = kb or BuildingKnowledgeBase()
        self._entry_cache: Dict[str, KnowledgeEntry] = {}

    def _find_entry(self, building: str) -> Optional[KnowledgeEntry]:
        if building in self._entry_cache:
            return self._entry_cache[building]
        hits = self.kb.search(building, top_k=1)
        if hits:
            self._entry_cache[building] = hits[0]
            return hits[0]
        return None

    @staticmethod
    def _field_text(entry: KnowledgeEntry, field: str) -> str:
        if field == "description":
            return entry.description or ""
        val = entry.metadata.get(field)
        if val is None:
            return ""
        if field == "height_m":
            return f"高约{val}米"
        if field == "built_year":
            return f"于{val}年建成"
        if field == "architect":
            return f"由{val}设计"
        if field == "designer":
            return f"主创设计师为{val}"
        if field == "location":
            return f"位于{val}"
        if field == "function":
            return f"主要功能为{val}"
        if field == "style":
            return val
        if field == "type":
            return f"建筑类型为{val}"
        if field == "city":
            return f"位于{val}"
        return str(val)

    def _match_intent(self, question: str) -> Optional[tuple]:
        q = question.strip()
        for keywords, fields in _INTENTS:
            if any(k in q for k in keywords):
                return fields
        return None

    def _compose(self, entry: KnowledgeEntry, fields: tuple, question: str) -> str:
        parts: List[str] = []
        name = entry.name
        for f in fields:
            if f == "description" and parts:
                # 已有针对性字段时，简介只取首句
                desc = (entry.description or "").split("，")[0]
                if desc and desc not in "".join(parts):
                    parts.append(desc)
                continue
            t = self._field_text(entry, f)
            if t and t not in "".join(parts):
                parts.append(t)
        if not parts:
            parts.append(entry.description or f"{name}是北京CBD地标建筑。")
        body = "，".join(parts[:2])
        if not body.endswith("。"):
            body += "。"
        return f"{name}：{body}"

    def _keyword_answer(self, entry: KnowledgeEntry, question: str) -> str:
        """按问题关键词从全部字段中检索"""
        q = question.strip()
        snippets: List[str] = []
        for field in (
            "description", "designer", "architect", "style", "function",
            "location", "height_m", "built_year", "type",
        ):
            text = self._field_text(entry, field)
            kws = re.split(r"[，。；？?！!、\s]+", q)
            if any(len(k) >= 2 and k in text for k in kws):
                snippets.append(text)
        if snippets:
            return f"{entry.name}：" + "，".join(dict.fromkeys(snippets)) + "。"
        return self._compose(entry, ("description", "function", "style"), q)

    def _facade_answer(self, entry: KnowledgeEntry, question: str) -> str:
        """外墙/玻璃/幕墙类问句 — 优先本地结构化字段作答"""
        q = question.strip()
        if not any(k in q for k in ("外墙", "玻璃", "幕墙", "外立面", "立面", "材料")):
            return ""
        style = self._field_text(entry, "style")
        desc = entry.description or ""
        facade_hint = style if any(w in style for w in ("玻璃", "幕墙")) else desc
        if not any(w in facade_hint for w in ("玻璃", "幕墙")):
            return ""

        yes_no = any(w in q for w in ("吗", "是不是", "是否", "全是", "都是", "全部"))
        if yes_no:
            return (
                f"{entry.name}：外立面以玻璃幕墙为主体（{style.rstrip('。')}），"
                f"但并非整栋楼每一面都 100% 纯玻璃覆盖，部分基座与结构区域仍有金属或石材包覆。"
            )
        return f"{entry.name}：{style.rstrip('。')}，{desc.split('，')[0]}。"

    def answer(
        self,
        building: str,
        question: str,
        on_progress: ProgressFn = None,
    ) -> str:
        if not question.strip():
            return "请输入问题"
        if on_progress:
            on_progress(0.15, "检索本地知识库")

        entry = self._find_entry(building)
        if not entry:
            if on_progress:
                on_progress(1.0, "完成")
            return f"本地知识库暂无「{building}」的详细资料。"

        if on_progress:
            on_progress(0.55, "匹配问题意图")

        facade_ans = self._facade_answer(entry, question)
        if facade_ans:
            if on_progress:
                on_progress(1.0, "完成")
            return facade_ans

        intent_fields = self._match_intent(question)
        if intent_fields:
            ans = self._compose(entry, intent_fields, question)
        else:
            ans = self._keyword_answer(entry, question)

        if on_progress:
            on_progress(0.9, "整理回答")
            on_progress(1.0, "完成")

        logger.info("本地问答: %s / %s -> %d字", building, question, len(ans))
        return ans
