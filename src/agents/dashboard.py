"""主终端看板 — 实时状态刷新"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class DashboardEntry:
    timestamp: float
    category: str   # System / Parallel / Handover / Agent
    message: str


class AgentDashboard:
    """主终端 Agent 状态看板"""

    def __init__(self, on_update: Optional[Callable[[str], None]] = None):
        self._lock = threading.Lock()
        self._entries: List[DashboardEntry] = []
        self._agent_status: Dict[str, str] = {}
        self._progress: float = 0.0
        self._on_update = on_update

    def log(self, category: str, message: str):
        with self._lock:
            self._entries.append(DashboardEntry(time.time(), category, message))
        self._emit()

    def set_agent_status(self, agent_id: str, status: str):
        with self._lock:
            self._agent_status[agent_id] = status
        self._emit()

    def set_progress(self, pct: float):
        with self._lock:
            self._progress = max(0.0, min(100.0, pct))
        self._emit()

    def reset_agents(self):
        with self._lock:
            for k in self._agent_status:
                self._agent_status[k] = "IDLE"

    def render(self) -> str:
        with self._lock:
            lines: List[str] = []
            lines.append(f"▓▓▓ 总体进度: {self._progress:.0f}% ▓▓▓")
            if self._agent_status:
                status_line = " | ".join(
                    f"● {aid}: {st}" for aid, st in sorted(self._agent_status.items())
                )
                lines.append(status_line)
            for e in self._entries[-30:]:
                prefix = f"[{e.category}]"
                lines.append(f"{prefix} {e.message}")
            return "\n".join(lines)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._agent_status.clear()
            self._progress = 0.0

    def _emit(self):
        if self._on_update:
            self._on_update(self.render())
