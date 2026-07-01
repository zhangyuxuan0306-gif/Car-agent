"""Critic Agent — 合规质检智能体"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

from src.agents.dashboard import AgentDashboard
from src.agents.schemas import CriticResult

logger = logging.getLogger(__name__)

_BANNED_AD = re.compile(
    r"(必吃|全网最低|史上最强|第一好吃|立即下单|限时优惠|扫码关注)"
)
_YEAR_PATTERN = re.compile(r"(19|20)\d{2}年?")


class CriticAgent:
    """检查导游文案是否存在幻觉、错误事实或不合规广告词"""

    def __init__(self, dashboard: AgentDashboard):
        self.dashboard = dashboard

    def verify(
        self,
        script: str,
        buildings: List[str],
        rag_context: Dict[str, str],
    ) -> CriticResult:
        self.dashboard.log("Handover", "Critic_Agent 正在进行合规性审查...")
        issues: List[str] = []

        if not script.strip():
            issues.append("文案为空")
            return CriticResult(passed=False, feedback="文案为空，请重新生成", issues=issues)

        # 广告词检查
        ad_match = _BANNED_AD.search(script)
        if ad_match:
            issues.append(f"含不合规广告词「{ad_match.group()}」")

        # 建筑名称是否出现在资料中
        known_names = set(buildings)
        for name in buildings:
            if name not in script and name[:2] not in script:
                issues.append(f"文案未提及识别建筑「{name}」")

        # 年份幻觉：文案中的年份须与知识库一致
        kb_years = self._collect_kb_years(rag_context)
        script_years = set(int(y.rstrip("年")) for y in _YEAR_PATTERN.findall(script))
        for y in script_years:
            if kb_years and y not in kb_years:
                # 允许 ±5 年容差（避免过度误杀）
                if not any(abs(y - ky) <= 5 for ky in kb_years):
                    issues.append(f"年份 {y} 与知识库不符（已知: {sorted(kb_years)}）")

        # 杜撰地址模式（常见幻觉）
        if re.search(r"位于(上海|广州|深圳|杭州)", script) and any(
            "北京" in ctx for ctx in rag_context.values()
        ):
            issues.append("地址可能错误：资料为北京建筑，文案出现其他城市")

        if issues:
            feedback = "；".join(issues)
            self.dashboard.log("Handover", f"Critic_Agent 未通过: {feedback}")
            return CriticResult(passed=False, feedback=feedback, issues=issues)

        self.dashboard.log("Handover", "Critic_Agent 审查通过 → PASSED")
        return CriticResult(passed=True, feedback="PASSED")

    @staticmethod
    def _collect_kb_years(rag_context: Dict[str, str]) -> Set[int]:
        years: Set[int] = set()
        for ctx in rag_context.values():
            for m in re.finditer(r"built_year:\s*(\d{4})", ctx):
                years.add(int(m.group(1)))
            for m in re.finditer(r"(19|20)\d{2}", ctx):
                years.add(int(m.group()))
        return years
