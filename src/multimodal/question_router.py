"""问答意图路由 — 先判断问的是什么，再决定走记忆 / 本地 / 联网"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# 对话记忆 / 元问题（不应检索建筑知识库或联网）
_MEMORY_PATTERNS = (
    "上个问题", "上一个问题", "上一个问题", "刚才问", "刚才的问题",
    "之前问", "之前的问题", "问过什么", "问过哪些", "我们聊", "聊过什么",
    "聊了什么", "还记得", "记得吗", "说过什么", "上一句", "上句话",
    "刚才说", "刚才说了", "最后一个问题", "对话历史", "之前说了",
    "你记得", "你还记得", "我问什么", "我问了", "之前讨论",
)

# 简单寒暄（仅对话历史 + 可选 LLM，不检索建筑资料）
_CHITCHAT_PATTERNS = (
    "你好", "您好", "谢谢", "感谢", "再见", "你是谁", "你能做什么",
    "帮帮我", "在吗", "早上好", "晚上好",
)

# 周边 POI / 生活服务（必须联网，不走建筑知识库）
_NEARBY_PATTERNS = (
    "附近", "周边", "周围", "旁边", "附近有", "周边有", "哪里有",
    "哪家", "有没有", "推荐", "好吃的", "好玩的",
    "咖啡厅", "咖啡馆", "咖啡", "奶茶", "茶饮",
    "餐厅", "饭店", "饭馆", "吃饭", "美食", "小吃",
    "停车", "停车场", "地铁", "公交站", "怎么去", "怎么走",
    "超市", "便利店", "商场", "购物", "酒店", "书店",
)

_POI_TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("咖啡厅", ("咖啡厅", "咖啡馆", "咖啡", "拿铁", "星巴克", "瑞幸")),
    ("餐厅", ("餐厅", "饭店", "饭馆", "吃饭", "美食", "小吃", "餐")),
    ("停车场", ("停车", "停车场", "车位")),
    ("交通", ("地铁", "公交", "怎么去", "怎么走", "路线")),
    ("购物", ("超市", "便利店", "商场", "购物")),
    ("酒店", ("酒店", "宾馆", "住宿")),
)


def poi_topic_label(question: str) -> str:
    for label, kws in _POI_TOPIC_KEYWORDS:
        if any(k in question for k in kws):
            return label
    return "相关地点"


def build_poi_queries(building: str, question: str, location: str = "") -> List[str]:
    """构造周边 POI 检索词"""
    topic = poi_topic_label(question)
    q_clean = re.sub(r"[？?！!。]", "", question.strip())
    queries: List[str] = []

    if location:
        queries.append(f"{location} {topic}")
        for part in re.split(r"[，,、]", location):
            part = part.strip()
            if len(part) >= 3 and part not in building:
                queries.append(f"{part} {topic}")

    queries.extend([
        f"北京 {building} 附近 {topic}",
        f"{building} 附近 {topic}",
        f"{building} 周边 {topic}",
        f"北京朝阳公园 {topic}",
        f"北京CBD {topic}",
        q_clean if building in q_clean else f"北京 {building} {q_clean}",
    ])

    seen: set[str] = set()
    out: List[str] = []
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def poi_keywords(question: str) -> tuple[str, ...]:
    kws = [k for label, group in _POI_TOPIC_KEYWORDS for k in group if k in question]
    if not kws:
        kws = [w for w in re.split(r"[，。；？?！!、\s]+", question) if len(w) >= 2]
    return tuple(dict.fromkeys(kws))


def poi_fallback_answer(building: str, question: str) -> str:
    topic = poi_topic_label(question)
    return (
        f"暂未检索到「{building}」周边{topic}的实时信息"
        f"（可能受网络限制或搜索验证影响）。"
        f"建议打开手机地图，搜索「{building} {topic}」查看最新结果。"
    )


def poi_offline_hint(building: str, question: str, location: str = "") -> str:
    """联网不可用时的区域级参考（不捏造具体店名）"""
    topic = poi_topic_label(question)
    area = location or "北京CBD"
    hints = {
        "咖啡厅": "商务区与朝阳公园周边通常有星巴克、瑞幸、M Stand 等连锁咖啡",
        "餐厅": "国贸、建外 SOHO、朝阳公园南门等片区餐饮选择较多",
        "停车场": "综合体及商场地下停车场较多，路边停车位紧张",
        "交通": "可乘坐地铁 1 号线国贸站、10 号线金台夕照站等",
        "购物": "国贸商城、世贸天阶等商圈较近",
        "酒店": "CBD 核心区内有多家商务酒店",
    }
    hint = hints.get(topic)
    if not hint:
        return poi_fallback_answer(building, question)
    return (
        f"「{building}」位于{area}。{hint}。"
        f"如需具体门店名单，建议打开地图 App 搜索「{building} {topic}」。"
    )


def classify_question(question: str) -> str:
    """
    返回问题类型：
    - memory  : 询问对话记忆（上个问题、刚才说了什么等）
    - chitchat: 寒暄 / 系统类
    - nearby  : 周边 POI / 生活服务（仅联网）
    - building: 建筑知识问答（走本地 RAG / 联网）
    """
    q = question.strip()
    if not q:
        return "building"
    if any(p in q for p in _MEMORY_PATTERNS):
        return "memory"
    if any(p in q for p in _CHITCHAT_PATTERNS) and len(q) <= 20:
        return "chitchat"
    if any(p in q for p in _NEARBY_PATTERNS):
        return "nearby"
    return "building"


def _user_messages(history: List[Tuple[str, str]], current: str) -> List[str]:
    """提取历史中的用户提问，排除当前这条（已在 hub 中写入）"""
    users = [content.strip() for role, content in history if role == "user" and content.strip()]
    if users and users[-1] == current.strip():
        users = users[:-1]
    return users


def _assistant_messages(history: List[Tuple[str, str]], current: str) -> List[str]:
    msgs = [content.strip() for role, content in history if role == "assistant" and content.strip()]
    return msgs


def answer_from_memory(
    question: str,
    history: List[Tuple[str, str]],
    building: str = "",
) -> Optional[str]:
    """基于对话历史直接回答元问题；无法回答时返回 None"""
    q = question.strip()
    users = _user_messages(history, q)
    assistants = _assistant_messages(history, q)

    if any(k in q for k in ("上个问题", "上一个问题", "上一个问题", "最后一个问题", "刚才问", "之前问")):
        if not users:
            return "这是我们本轮对话里的第一个问题，之前还没有其它提问。"
        return f"您上一个问题是：「{users[-1]}」"

    if any(k in q for k in ("上一句", "上句话", "刚才说", "刚才说了", "之前说了")):
        if assistants:
            return f"我上一句回答是：「{assistants[-1][:200]}{'…' if len(assistants[-1]) > 200 else ''}」"
        if users:
            return f"我还没有正式回答，您刚才问的是：「{users[-1]}」"
        return "目前还没有可回顾的对话内容。"

    if any(k in q for k in ("聊过什么", "聊了什么", "说过什么", "问过什么", "问过哪些", "我们聊", "对话历史")):
        if not users and not assistants:
            return "目前还没有对话记录。"
        lines = ["回顾本次对话："]
        pairs = [(r, c) for r, c in history if r in ("user", "assistant")][-10:]
        for role, content in pairs:
            prefix = "您" if role == "user" else "我"
            lines.append(f"- {prefix}：{content[:120]}{'…' if len(content) > 120 else ''}")
        return "\n".join(lines)

    if any(k in q for k in ("还记得", "记得吗", "你记得", "你还记得")):
        if not history:
            return f"目前我们刚开始聊{'「' + building + '」' if building else ''}，还没有太多可以回忆的内容。"
        summary = users[-3:] if users else []
        if summary:
            joined = "；".join(f"「{u}」" for u in summary)
            return f"记得的。您最近问过：{joined}。"
        return "记得我们刚才的开场介绍，您可以继续提问。"

    return None
