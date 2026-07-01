#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能座舱多智能体协同框架 — 单文件可运行演示
基于 asyncio + rich 实现 TUI 看板、后台静默感知、车载压力测试。

运行：python smart_cockpit_agents.py
依赖：pip install rich
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from rich import box
from rich.columns import Columns
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# ═══════════════════════════════════════════════════════════════════════════════
# 一、数据结构 & 感知缓存
# ═══════════════════════════════════════════════════════════════════════════════

# 窗外三区预置地标（演示用）
REGION_ENTITIES = {
    "left":   {"name": "景山万春亭",   "bbox": (0.05, 0.15, 0.28, 0.55)},
    "center": {"name": "全聚德烤鸭店", "bbox": (0.35, 0.20, 0.62, 0.58)},
    "right":  {"name": "联想总部大楼", "bbox": (0.68, 0.12, 0.95, 0.60)},
}


class AgentState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    RETRY = "RETRY"


@dataclass
class EntityPerception:
    """单目标视觉识别结果"""
    name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    region: str
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class HandoverJSON:
    """Agent 间结构化交接协议"""
    phase: str
    click: Tuple[int, int] = (0, 0)
    entity: Optional[EntityPerception] = None
    rag_context: str = ""
    draft_script: str = ""
    critic_feedback: str = ""
    final_script: str = ""
    cache_hit: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        e = self.entity
        return {
            "phase": self.phase,
            "click": list(self.click),
            "entity": None if not e else {
                "name": e.name, "bbox": list(e.bbox),
                "confidence": e.confidence, "region": e.region,
            },
            "rag_context": self.rag_context[:80] + "…" if len(self.rag_context) > 80 else self.rag_context,
            "final_script": self.final_script,
            "cache_hit": self.cache_hit,
            "latency_ms": round(self.latency_ms, 1),
        }


class PerceptionCache:
    """后台静默感知缓存 — 左/中/右三区实体索引"""

    def __init__(self):
        self._regions: Dict[str, EntityPerception] = {}
        self._writes = 0
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()

    async def write(self, entity: EntityPerception) -> None:
        async with self._lock:
            self._regions[entity.region] = entity
            self._writes += 1

    async def lookup(self, x: int, y: int, width: int = 1000, height: int = 600) -> Optional[EntityPerception]:
        """根据点击坐标命中缓存区域（归一化 bbox）"""
        async with self._lock:
            nx, ny = x / width, y / height
            best: Optional[EntityPerception] = None
            best_dist = float("inf")
            for ent in self._regions.values():
                x1, y1, x2, y2 = ent.bbox
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    self._hits += 1
                    return ent
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d = (cx - nx) ** 2 + (cy - ny) ** 2
                if d < best_dist:
                    best_dist, best = d, ent
            if best and best_dist < 0.08:
                self._hits += 1
                return best
            self._misses += 1
            return None

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total * 100) if total else 0.0

    @property
    def write_count(self) -> int:
        return self._writes

    def snapshot(self) -> Dict[str, str]:
        return {r: e.name for r, e in self._regions.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 二、TUI 看板
# ═══════════════════════════════════════════════════════════════════════════════

class Dashboard:
    """rich.live 双栏看板：左状态 / 右日志"""

    def __init__(self):
        self.global_progress: float = 0.0
        self.phase_label: str = "初始化中"
        self.test_mode: str = "—"
        self.agent_status: Dict[str, AgentState] = {
            "Master": AgentState.IDLE,
            "Vision_Agent_1": AgentState.IDLE,
            "Vision_Agent_2": AgentState.IDLE,
            "Vision_Agent_3": AgentState.IDLE,
            "RAG_Agent": AgentState.IDLE,
            "Writer_Agent": AgentState.IDLE,
            "Critic_Agent": AgentState.IDLE,
        }
        self.model_ready: bool = False
        self.cache_snapshot: Dict[str, str] = {}
        self.logs: Deque[str] = deque(maxlen=40)
        self.metrics: Dict[str, str] = {}

    def log(self, category: str, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[dim]{ts}[/dim] [{category}] {msg}")

    def set_agent(self, agent_id: str, state: AgentState) -> None:
        self.agent_status[agent_id] = state

    def render(self) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3),
        )

        # ── 左侧：进度 + 状态 ──
        prog = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        task_id = prog.add_task(self.phase_label, total=100, completed=self.global_progress)

        agent_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("状态")
        color_map = {
            AgentState.IDLE: "dim",
            AgentState.RUNNING: "yellow",
            AgentState.DONE: "green",
            AgentState.FAILED: "red",
            AgentState.RETRY: "orange1",
        }
        for aid, st in self.agent_status.items():
            agent_table.add_row(aid, f"[{color_map[st]}]● {st.value}[/]")

        cache_lines = "\n".join(
            f"  • {r}: [bold]{n}[/]" for r, n in self.cache_snapshot.items()
        ) or "  （缓存空）"

        model_txt = "[green]✓ VLM/LLM 已常驻显存[/]" if self.model_ready else "[yellow]⏳ 模型加载中…[/]"

        left_content = Group(
            Panel(prog, title="[bold]全局进度[/bold]", border_style="blue"),
            Panel(
                f"测试路况：[bold yellow]{self.test_mode}[/]\n"
                f"模型状态：{model_txt}\n\n"
                f"[bold]PerceptionCache 快照：[/]\n{cache_lines}",
                title="系统状态",
                border_style="green",
            ),
            Panel(agent_table, title="Agent 状态机", border_style="magenta"),
        )
        if self.metrics:
            mt = Table(box=box.SIMPLE)
            mt.add_column("指标", style="cyan")
            mt.add_column("数值")
            for k, v in self.metrics.items():
                mt.add_row(k, v)
            left_content = Group(left_content, Panel(mt, title="压力测试指标", border_style="yellow"))

        layout["left"].update(Panel(left_content, title="🏙️ 智能座舱 Master 终端", border_style="bright_blue"))

        # ──右侧：交接日志 ──
        log_text = Text.from_markup("\n".join(reversed(self.logs))) if self.logs else Text("等待日志…", style="dim")
        layout["right"].update(
            Panel(log_text, title="📡 多智能体交接日志（实时滚动）", border_style="bright_cyan")
        )
        return layout


# ═══════════════════════════════════════════════════════════════════════════════
# 三、各 Agent 实现（异步模拟）
# ═══════════════════════════════════════════════════════════════════════════════

# 知识库（模拟 RAG 本地检索）
KNOWLEDGE_DB: Dict[str, str] = {
    "景山万春亭": "建于1751年，景山最高点，可俯瞰故宫与中轴线，明清皇家御苑核心建筑。",
    "全聚德烤鸭店": "中华老字号，1864年创立，挂炉烤鸭技艺列入非物质文化遗产。",
    "联想总部大楼": "联想集团北京总部，位于中关村，现代玻璃幕墙科技园区地标。",
}

# 违规词（Critic 检测）
_BANNED = ("必吃", "全网最低", "第一好吃")


class VisionAgent:
    """视觉识别 Agent — 负责单区域静默感知"""

    def __init__(self, agent_id: str, region: str, dashboard: Dashboard):
        self.agent_id = agent_id
        self.region = region
        self.dashboard = dashboard

    async def perceive(
        self,
        cache: PerceptionCache,
        *,
        fail_rate: float = 0.0,
        latency_ms: Tuple[float, float] = (40, 120),
    ) -> bool:
        """识别窗外指定区域，写入 PerceptionCache；fail_rate 模拟高速丢帧"""
        self.dashboard.set_agent(self.agent_id, AgentState.RUNNING)
        await asyncio.sleep(random.uniform(*latency_ms) / 1000)

        if random.random() < fail_rate:
            self.dashboard.set_agent(self.agent_id, AgentState.FAILED)
            self.dashboard.log("后台静默", f"{self.agent_id} 区域[{self.region}] 识别失败（车速过快/模糊）")
            return False

        meta = REGION_ENTITIES[self.region]
        entity = EntityPerception(
            name=meta["name"],
            bbox=meta["bbox"],
            confidence=round(random.uniform(0.82, 0.98), 3),
            region=self.region,
        )
        await cache.write(entity)
        self.dashboard.set_agent(self.agent_id, AgentState.DONE)
        self.dashboard.log(
            "后台静默",
            f"{self.agent_id} 识别成功 → 【{entity.name}】 conf={entity.confidence:.2f} → 已写入缓存",
        )
        return True


class RAGAgent:
    async def retrieve(self, entity: EntityPerception, dashboard: Dashboard) -> str:
        dashboard.set_agent("RAG_Agent", AgentState.RUNNING)
        dashboard.log("Handover", f"RAG_Agent 检索【{entity.name}】历史与背景数据…")
        await asyncio.sleep(random.uniform(0.08, 0.18))
        ctx = KNOWLEDGE_DB.get(entity.name, f"{entity.name}：暂无详细资料。")
        # 模拟联网补全
        ctx += "\n【联网】大众点评/百科摘要已并入上下文。"
        dashboard.set_agent("RAG_Agent", AgentState.DONE)
        dashboard.log("Handover", f"RAG_Agent 检索完成，上下文 {len(ctx)} 字")
        return ctx


class WriterAgent:
    async def write(
        self,
        entity: EntityPerception,
        rag_context: str,
        dashboard: Dashboard,
        feedback: str = "",
    ) -> str:
        dashboard.set_agent("Writer_Agent", AgentState.RUNNING)
        if feedback:
            dashboard.log("Handover", f"Writer_Agent 收到 Critic 反馈，重写：{feedback[:40]}")
        else:
            dashboard.log("Handover", "Writer_Agent 生成车载导游播报词…")
        await asyncio.sleep(random.uniform(0.06, 0.14))
        script = (
            f"乘客您好，您窗外正在经过的是{entity.name}。"
            f"{rag_context.split('。')[0]}。"
            f"如需了解更多，可随时语音提问。"
        )
        if "必吃" in script:
            script = script.replace("必吃", "值得品尝")
        dashboard.set_agent("Writer_Agent", AgentState.DONE)
        return script


class CriticAgent:
    async def verify(self, script: str, entity: EntityPerception, dashboard: Dashboard) -> Tuple[bool, str]:
        dashboard.set_agent("Critic_Agent", AgentState.RUNNING)
        dashboard.log("Handover", "Critic_Agent 合规性审查中…")
        await asyncio.sleep(random.uniform(0.04, 0.10))
        issues = [w for w in _BANNED if w in script]
        if issues:
            fb = f"含不合规词：{'、'.join(issues)}"
            dashboard.set_agent("Critic_Agent", AgentState.FAILED)
            dashboard.log("Handover", f"Critic_Agent 未通过 → {fb}")
            return False, fb
        dashboard.set_agent("Critic_Agent", AgentState.DONE)
        dashboard.log("Handover", "Critic_Agent 审查通过 → PASSED")
        return True, "PASSED"


# ═══════════════════════════════════════════════════════════════════════════════
# 四、Master 编排器
# ═══════════════════════════════════════════════════════════════════════════════

class Master:
    """主终端 — 模型预加载、后台静默感知、点击触发异构协作"""

    def __init__(self, dashboard: Dashboard, cache: PerceptionCache):
        self.dashboard = dashboard
        self.cache = cache
        self.vision_agents = [
            VisionAgent("Vision_Agent_1", "left", dashboard),
            VisionAgent("Vision_Agent_2", "center", dashboard),
            VisionAgent("Vision_Agent_3", "right", dashboard),
        ]
        self.rag = RAGAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()
        self._bg_running = False
        self._bg_task: Optional[asyncio.Task] = None

    async def preload_models(self) -> None:
        """阶段 0：模拟 VLM/LLM 参数常驻显存"""
        self.dashboard.phase_label = "模型预加载中"
        self.dashboard.set_agent("Master", AgentState.RUNNING)
        self.dashboard.log("System", "Master 启动，VLM/LLM 参数加载至显存…")
        for pct in range(0, 101, 10):
            self.dashboard.global_progress = pct
            await asyncio.sleep(0.12)
        self.dashboard.model_ready = True
        self.dashboard.log("System", "模型预加载完成，进入后台静默感知模式")
        self.dashboard.set_agent("Master", AgentState.DONE)

    async def background_perceive_loop(
        self,
        *,
        interval: float = 0.35,
        fail_rate: float = 0.0,
        burst: int = 1,
    ) -> None:
        """后台持续并发调度 Vision×3（核心创新点 1）"""
        self._bg_running = True
        self.dashboard.log("后台静默", "视觉团队开始并发感知左/中/右三区…")
        while self._bg_running:
            self.dashboard.set_agent("Master", AgentState.RUNNING)
            for va in self.vision_agents:
                self.dashboard.set_agent(va.agent_id, AgentState.IDLE)

            tasks = [
                va.perceive(self.cache, fail_rate=fail_rate)
                for va in self.vision_agents
            ]
            if burst > 1:
                tasks = tasks * burst  # 密集商业街：加倍并发写入

            await asyncio.gather(*tasks)
            self.dashboard.cache_snapshot = self.cache.snapshot()
            self.dashboard.set_agent("Master", AgentState.IDLE)
            await asyncio.sleep(interval)

    def stop_background(self) -> None:
        self._bg_running = False

    async def on_click(self, x: int, y: int, *, max_critic_retry: int = 2) -> HandoverJSON:
        """被动点击 — 缓存命中 → RAG → Writer → Critic"""
        t0 = time.perf_counter()
        payload = HandoverJSON(phase="click", click=(x, y))
        self.dashboard.set_agent("Master", AgentState.RUNNING)
        self.dashboard.log("前台触发", f"捕获点击坐标 ({x}, {y})")

        entity = await self.cache.lookup(x, y)
        if entity:
            payload.cache_hit = True
            payload.entity = entity
            payload.phase = "cache_hit"
            self.dashboard.log(
                "前台触发",
                f"[bold green]命中缓存！[/] 瞬时唤醒 → 【{entity.name}】 0ms 视觉延迟",
            )
        else:
            payload.cache_hit = False
            self.dashboard.log("前台触发", "[yellow]缓存未命中，同步 fallback 识别…[/]")
            await asyncio.sleep(0.25)
            region = "center" if 0.33 < x / 1000 < 0.66 else ("left" if x / 1000 < 0.5 else "right")
            meta = REGION_ENTITIES[region]
            entity = EntityPerception(meta["name"], meta["bbox"], 0.75, region)
            payload.entity = entity

        # 异构协作链
        rag_ctx = await self.rag.retrieve(entity, self.dashboard)
        payload.rag_context = rag_ctx

        feedback = ""
        script = ""
        for attempt in range(max_critic_retry + 1):
            script = await self.writer.write(entity, rag_ctx, self.dashboard, feedback)
            payload.draft_script = script
            passed, feedback = await self.critic.verify(script, entity, self.dashboard)
            payload.critic_feedback = feedback
            if passed:
                break
            if attempt < max_critic_retry:
                self.dashboard.set_agent("Writer_Agent", AgentState.RETRY)
                self.dashboard.log("Handover", f"文案退回 Writer（第 {attempt + 2} 次）")

        payload.final_script = script
        payload.phase = "done"
        payload.latency_ms = (time.perf_counter() - t0) * 1000
        self.dashboard.log(
            "System",
            f"导游词已推送车载大屏/TTS | 耗时 {payload.latency_ms:.0f}ms | 缓存命中={payload.cache_hit}",
        )
        self.dashboard.set_agent("Master", AgentState.DONE)
        return payload


# ═══════════════════════════════════════════════════════════════════════════════
# 五、车载压力测试（核心创新点 2）
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StressTestResult:
    name: str
    cache_hit_rate: float
    avg_latency_ms: float
    cache_writes: int
    samples: int
    notes: str


class AutomotiveStressTest:
    """
    三组车载极端路况模拟：
    A — 低速密集商业街（高吞吐缓存写入）
    B — 高速行驶（识别失败 + 重试容错）
    C — 常规即时触发（端到端超低延迟）
    """

    def __init__(self, master: Master, dashboard: Dashboard, cache: PerceptionCache):
        self.master = master
        self.dashboard = dashboard
        self.cache = cache
        self.results: List[StressTestResult] = []

    async def run_all(self) -> List[StressTestResult]:
        self.results.clear()
        await self._test_a_dense_street()
        await self._test_b_high_speed()
        await self._test_c_instant_trigger()
        return self.results

    async def _test_a_dense_street(self) -> None:
        """测试组 A：低速密集商业街"""
        self.dashboard.test_mode = "A · 低速密集商业街"
        self.dashboard.phase_label = "压力测试 A"
        self.dashboard.global_progress = 0
        self.dashboard.log("StressTest", "═══ 测试组 A：低速密集商业街 ═══")
        self.dashboard.log("StressTest", "高并发写入 PerceptionCache，观测吞吐…")

        writes_before = self.cache.write_count
        self.master.stop_background()
        await asyncio.sleep(0.1)

        # 5 轮 burst×3 并发
        for i in range(5):
            self.dashboard.global_progress = (i + 1) * 18
            tasks = [
                va.perceive(self.cache, fail_rate=0.05, latency_ms=(20, 60))
                for va in self.master.vision_agents
            ]
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.08)

        writes = self.cache.write_count - writes_before
        self.dashboard.cache_snapshot = self.cache.snapshot()

        # 模拟 10 次点击
        latencies: List[float] = []
        hits = 0
        clicks = [(150, 300), (500, 320), (820, 280), (480, 350), (200, 400),
                  (750, 300), (500, 250), (300, 380), (600, 300), (850, 350)]
        for cx, cy in clicks:
            p = await self.master.on_click(cx, cy)
            latencies.append(p.latency_ms)
            if p.cache_hit:
                hits += 1

        result = StressTestResult(
            name="A · 低速密集商业街",
            cache_hit_rate=hits / len(clicks) * 100,
            avg_latency_ms=sum(latencies) / len(latencies),
            cache_writes=writes,
            samples=len(clicks),
            notes=f"5轮×3并发写入，缓存写入 {writes} 次",
        )
        self.results.append(result)
        self.dashboard.log("StressTest", f"A 组完成 | 命中率 {result.cache_hit_rate:.0f}% | 均延迟 {result.avg_latency_ms:.0f}ms")

    async def _test_b_high_speed(self) -> None:
        """测试组 B：高速行驶 — 识别失败与重试"""
        self.dashboard.test_mode = "B · 高速行驶"
        self.dashboard.phase_label = "压力测试 B"
        self.dashboard.global_progress = 0
        self.dashboard.log("StressTest", "═══ 测试组 B：高速行驶路况 ═══")
        self.dashboard.log("StressTest", "模拟画面模糊，fail_rate=45%，测试容错…")

        fail_count = 0
        success_count = 0
        for i in range(8):
            self.dashboard.global_progress = (i + 1) * 12
            outcomes = await asyncio.gather(*[
                va.perceive(self.cache, fail_rate=0.45, latency_ms=(15, 40))
                for va in self.master.vision_agents
            ])
            fail_count += sum(1 for o in outcomes if not o)
            success_count += sum(1 for o in outcomes if o)
            await asyncio.sleep(0.05)

        # 失败后用 retry 补感知
        self.dashboard.log("StressTest", f"高速段失败 {fail_count} 次，触发补偿感知…")
        await asyncio.gather(*[
            va.perceive(self.cache, fail_rate=0.1, latency_ms=(30, 80))
            for va in self.master.vision_agents
        ])

        latencies: List[float] = []
        hits = 0
        for cx, cy in [(500, 300), (180, 280), (800, 320)]:
            p = await self.master.on_click(cx, cy)
            latencies.append(p.latency_ms)
            if p.cache_hit:
                hits += 1

        result = StressTestResult(
            name="B · 高速行驶",
            cache_hit_rate=hits / 3 * 100,
            avg_latency_ms=sum(latencies) / len(latencies),
            cache_writes=success_count,
            samples=3,
            notes=f"失败 {fail_count} 次后补偿感知，成功率 {success_count}/{fail_count + success_count}",
        )
        self.results.append(result)
        self.dashboard.log("StressTest", f"B 组完成 | 命中率 {result.cache_hit_rate:.0f}% | 容错后均延迟 {result.avg_latency_ms:.0f}ms")

    async def _test_c_instant_trigger(self) -> None:
        """测试组 C：常规即时触发 — 端到端延迟"""
        self.dashboard.test_mode = "C · 常规即时触发"
        self.dashboard.phase_label = "压力测试 C"
        self.dashboard.global_progress = 0
        self.dashboard.log("StressTest", "═══ 测试组 C：常规即时触发 ═══")

        # 预热缓存
        await asyncio.gather(*[
            va.perceive(self.cache, fail_rate=0.0, latency_ms=(10, 30))
            for va in self.master.vision_agents
        ])
        self.dashboard.cache_snapshot = self.cache.snapshot()

        latencies: List[float] = []
        hits = 0
        for cx, cy in [(500, 310), (200, 290), (850, 300)]:
            self.dashboard.global_progress += 30
            p = await self.master.on_click(cx, cy)
            latencies.append(p.latency_ms)
            if p.cache_hit:
                hits += 1
            self.dashboard.log("StressTest", f"  点击({cx},{cy}) → {p.latency_ms:.0f}ms 命中={p.cache_hit}")

        result = StressTestResult(
            name="C · 常规即时触发",
            cache_hit_rate=hits / 3 * 100,
            avg_latency_ms=sum(latencies) / len(latencies),
            cache_writes=3,
            samples=3,
            notes="预热缓存后纯端到端路径",
        )
        self.results.append(result)
        self.dashboard.global_progress = 100
        self.dashboard.log("StressTest", f"C 组完成 | 命中率 {result.cache_hit_rate:.0f}% | E2E {result.avg_latency_ms:.0f}ms")

    def print_summary(self, dashboard: Dashboard) -> None:
        """终端打印对比表"""
        dashboard.metrics = {
            r.name: f"命中 {r.cache_hit_rate:.0f}% | {r.avg_latency_ms:.0f}ms"
            for r in self.results
        }
        dashboard.log("StressTest", "──────── 压力测试汇总 ────────")
        for r in self.results:
            dashboard.log(
                "StressTest",
                f"[bold]{r.name}[/]  命中率={r.cache_hit_rate:.1f}%  "
                f"均延迟={r.avg_latency_ms:.1f}ms  样本={r.samples}  ({r.notes})",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 六、主程序入口
# ═══════════════════════════════════════════════════════════════════════════════

async def main_demo(live: Live, dashboard: Dashboard) -> None:
    cache = PerceptionCache()
    master = Master(dashboard, cache)
    stress = AutomotiveStressTest(master, dashboard, cache)

    # ── 阶段 0：模型预加载 ──
    await master.preload_models()
    live.update(dashboard.render())
    await asyncio.sleep(0.5)

    # ── 阶段 1：后台静默感知（演示 3 轮） ──
    dashboard.phase_label = "后台静默感知"
    dashboard.global_progress = 10
    dashboard.log("System", "用户未点击，Master 后台调度 Vision×3 并发…")
    live.update(dashboard.render())

    bg = asyncio.create_task(master.background_perceive_loop(interval=0.01, fail_rate=0.0))
    for i in range(3):
        dashboard.global_progress = 10 + i * 8
        live.update(dashboard.render())
        await asyncio.sleep(0.45)
    master.stop_background()
    bg.cancel()
    try:
        await bg
    except asyncio.CancelledError:
        pass

    dashboard.log("System", f"静默感知完成，缓存实体：{cache.snapshot()}")
    live.update(dashboard.render())
    await asyncio.sleep(0.4)

    # ── 阶段 2：模拟用户点击（演示） ──
    dashboard.phase_label = "前台点击协作"
    dashboard.global_progress = 35
    dashboard.log("System", "模拟乘客点击车窗坐标 (500, 300)…")
    live.update(dashboard.render())
    payload = await master.on_click(500, 300)
    dashboard.global_progress = 50
    dashboard.log("System", f"最终播报：{payload.final_script[:60]}…")
    live.update(dashboard.render())
    await asyncio.sleep(0.6)

    # ── 阶段 3：车载压力测试 ──
    dashboard.phase_label = "车载压力测试"
    dashboard.log("System", "启动 AutomotiveStressTest 三组极端路况…")
    live.update(dashboard.render())
    await stress.run_all()
    stress.print_summary(dashboard)
    dashboard.phase_label = "全部完成"
    dashboard.global_progress = 100
    dashboard.log("System", "[bold green]任务圆满完成！[/] 看板 100%")
    live.update(dashboard.render())
    await asyncio.sleep(2.5)


def main() -> None:
    dashboard = Dashboard()
    with Live(dashboard.render(), refresh_per_second=8, screen=False) as live:
        asyncio.run(main_demo(live, dashboard))


if __name__ == "__main__":
    main()
