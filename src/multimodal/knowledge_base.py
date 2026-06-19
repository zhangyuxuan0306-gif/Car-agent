"""建筑知识库 - 支持知识补全与检索增强"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE = [
    {
        "name": "东方明珠",
        "aliases": ["东方明珠塔", "Oriental Pearl Tower"],
        "city": "上海",
        "type": "电视塔",
        "built_year": 1994,
        "height_m": 468,
        "architect": "江欢成",
        "style": "后现代主义",
        "description": "东方明珠广播电视塔是上海的标志性文化景观之一，位于浦东新区陆家嘴，高468米。塔体由三个大小不一的球体串联而成，寓意'大珠小珠落玉盘'。",
        "features": ["空中旋转餐厅", "上海历史发展陈列馆", "太空舱观光层"],
        "cultural_value": "上海改革开放的象征，中国首批5A级旅游景区",
    },
    {
        "name": "故宫博物院",
        "aliases": ["故宫", "紫禁城", "Forbidden City"],
        "city": "北京",
        "type": "宫殿建筑群",
        "built_year": 1420,
        "area_sqm": 720000,
        "architect": "蒯祥等",
        "style": "明清宫殿建筑",
        "description": "故宫是中国明清两代的皇家宫殿，是世界上现存规模最大、保存最为完整的木质结构古建筑群，1987年被列为世界文化遗产。",
        "features": ["太和殿", "乾清宫", "御花园", "珍宝馆"],
        "cultural_value": "世界文化遗产，中国古代宫廷建筑之精华",
    },
    {
        "name": "上海中心大厦",
        "aliases": ["上海中心", "Shanghai Tower"],
        "city": "上海",
        "type": "超高层摩天大楼",
        "built_year": 2015,
        "height_m": 632,
        "description": "上海中心大厦高632米，是中国第一高楼、世界第三高楼，位于陆家嘴金融区，外形呈螺旋上升的圆柱形。",
    },
    {
        "name": "上海环球金融中心",
        "aliases": ["环球金融中心", "Shanghai World Financial Center", "开瓶器"],
        "city": "上海",
        "type": "超高层摩天大楼",
        "built_year": 2008,
        "height_m": 492,
        "description": "上海环球金融中心高492米，顶部梯形风洞是其标志特征，是陆家嘴天际线核心建筑。",
    },
    {
        "name": "金茂大厦",
        "aliases": ["金茂", "Jin Mao Tower"],
        "city": "上海",
        "type": "超高层摩天大楼",
        "built_year": 1999,
        "height_m": 420,
        "description": "金茂大厦高420.5米，融合中国古典塔楼比例与现代玻璃幕墙，是陆家嘴早期地标。",
    },
    {
        "name": "震旦国际大楼",
        "aliases": ["震旦", "Aurora Plaza"],
        "city": "上海",
        "type": "办公楼",
        "description": "震旦国际大楼位于陆家嘴，以其金色外观著称。",
    },
    {
        "name": "广州塔",
        "aliases": ["小蛮腰", "Canton Tower"],
        "city": "广州",
        "type": "电视塔",
        "built_year": 2010,
        "height_m": 600,
        "architect": "马克·海默尔",
        "style": "现代结构主义",
        "description": "广州塔昵称'小蛮腰'，塔身中部收窄，犹如少女的纤细腰肢。塔高600米，是中国第一高塔，世界第三高塔。",
        "features": ["摩天轮", "蜘蛛侠栈道", "极速云霄"],
        "cultural_value": "广州城市新地标，2010年亚运会重要设施",
    },
]


@dataclass
class KnowledgeEntry:
    """知识条目"""
    name: str
    aliases: List[str]
    city: str
    building_type: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_context(self) -> str:
        lines = [f"建筑名称: {self.name}"]
        if self.city:
            lines.append(f"所在城市: {self.city}")
        if self.building_type:
            lines.append(f"建筑类型: {self.building_type}")
        for key, val in self.metadata.items():
            if key not in ("name", "aliases", "city", "type", "description"):
                lines.append(f"{key}: {val}")
        lines.append(f"简介: {self.description}")
        return "\n".join(lines)


class BuildingKnowledgeBase:
    """建筑知识库"""

    def __init__(self, db_path: str = "data/knowledge/buildings.json"):
        self.db_path = db_path
        self.entries: List[KnowledgeEntry] = []
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = DEFAULT_KNOWLEDGE
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        self.entries = []
        for item in data:
            self.entries.append(KnowledgeEntry(
                name=item.get("name", ""),
                aliases=item.get("aliases", []),
                city=item.get("city", ""),
                building_type=item.get("type", ""),
                description=item.get("description", ""),
                metadata={k: v for k, v in item.items()
                          if k not in ("name", "aliases", "city", "type", "description")},
            ))
        logger.info("知识库已加载 %d 条建筑记录", len(self.entries))

    def search(self, query: str, top_k: int = 3) -> List[KnowledgeEntry]:
        """简单关键词检索"""
        query_lower = query.lower()
        scored = []
        for entry in self.entries:
            score = 0
            if query_lower in entry.name.lower():
                score += 10
            for alias in entry.aliases:
                if query_lower in alias.lower():
                    score += 8
            if query_lower in entry.city.lower():
                score += 3
            if query_lower in entry.building_type.lower():
                score += 2
            if query_lower in entry.description.lower():
                score += 1
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def get_context_for_building(self, building_label: str, vlm_description: str = "") -> str:
        """为建筑目标获取知识补全上下文"""
        results = self.search(building_label, top_k=1)
        if results:
            return results[0].to_context()

        # 尝试从 VLM 描述中检索
        if vlm_description:
            for entry in self.entries:
                for alias in [entry.name] + entry.aliases:
                    if alias in vlm_description:
                        return entry.to_context()

        return ""

    def add_entry(self, entry: KnowledgeEntry):
        """动态添加知识条目"""
        self.entries.append(entry)
        self._save()

    def _save(self):
        data = []
        for e in self.entries:
            item = {
                "name": e.name,
                "aliases": e.aliases,
                "city": e.city,
                "type": e.building_type,
                "description": e.description,
                **e.metadata,
            }
            data.append(item)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
