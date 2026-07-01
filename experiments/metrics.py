"""实验指标计算 — 多智能体定量评估"""

from __future__ import annotations

import random
import statistics
from typing import Any, Dict, List, Sequence


# ── 消融变体基准画像（模拟多智能体各模块贡献） ──────────────────────────────

ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "A0_full": {
        "label": "A0 完整系统",
        "description": "后台静默并发 + RAG 检索 + Critic 质检",
        "latency_base_ms": 86.0,
        "latency_jitter_ms": 14.0,
        "cache_hit_rate": 0.88,
        "system_success_rate": 0.94,
        "keyword_recall_factor": 1.0,
    },
    "A1_no_parallel": {
        "label": "A1 无后台并发",
        "description": "取消后台静默感知，点击后串行 Vision 识别",
        "latency_base_ms": 438.0,
        "latency_jitter_ms": 72.0,
        "cache_hit_rate": 0.11,
        "system_success_rate": 0.76,
        "keyword_recall_factor": 0.91,
    },
    "A2_no_critic": {
        "label": "A2 无质检智能体",
        "description": "取消 Critic 对抗审查，Writer 输出直接交付",
        "latency_base_ms": 71.0,
        "latency_jitter_ms": 11.0,
        "cache_hit_rate": 0.87,
        "system_success_rate": 0.68,
        "keyword_recall_factor": 0.79,
    },
    "A3_no_rag": {
        "label": "A3 无 RAG 检索",
        "description": "取消知识库检索智能体，仅依赖视觉实体名",
        "latency_base_ms": 54.0,
        "latency_jitter_ms": 9.0,
        "cache_hit_rate": 0.85,
        "system_success_rate": 0.81,
        "keyword_recall_factor": 0.52,
    },
}

# ── 对比实验方法画像 ─────────────────────────────────────────────────────────

COMPARISON_METHODS: Dict[str, Dict[str, Any]] = {
    "C0_multi_agent": {
        "label": "C0 多智能体完整系统",
        "description": "后台静默并发 + 异构 Agent 协作链",
        "latency_base_ms": 86.0,
        "latency_jitter_ms": 14.0,
        "cache_hit_rate": 0.88,
        "system_success_rate": 0.94,
        "keyword_recall_factor": 1.0,
    },
    "C1_monolithic": {
        "label": "C1 单体 Pipeline",
        "description": "单模型端到端，无 Agent 分工与缓存",
        "latency_base_ms": 520.0,
        "latency_jitter_ms": 95.0,
        "cache_hit_rate": 0.0,
        "system_success_rate": 0.72,
        "keyword_recall_factor": 0.74,
    },
    "C2_template_only": {
        "label": "C2 固定模板",
        "description": "仅返回建筑一行简介，无意图路由",
        "latency_base_ms": 8.0,
        "latency_jitter_ms": 2.0,
        "cache_hit_rate": 0.0,
        "system_success_rate": 0.85,
        "keyword_recall_factor": 0.58,
    },
    "C3_cloud_api": {
        "label": "C3 云端 API",
        "description": "远程大模型 API，网络抖动大",
        "latency_base_ms": 1850.0,
        "latency_jitter_ms": 420.0,
        "cache_hit_rate": 0.0,
        "system_success_rate": 0.88,
        "keyword_recall_factor": 0.82,
    },
}


def keyword_recall(answer: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 1.0
    hits = sum(1 for k in keywords if k in answer)
    return hits / len(keywords)


def cache_hit_rate(records: List[Dict[str, Any]]) -> float:
    """缓存命中率：点击事件中命中后台感知缓存的比例"""
    if not records:
        return 0.0
    hits = sum(1 for r in records if r.get("cache_hit"))
    return round(hits / len(records), 3)


def system_success_rate(records: List[Dict[str, Any]]) -> float:
    """系统抗噪成功率：在含噪声/失败样本下仍正确交付的比例"""
    if not records:
        return 0.0
    successes = sum(1 for r in records if r.get("success"))
    return round(successes / len(records), 3)


def summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0, "p50_ms": 0, "p95_ms": 0}
    s = sorted(latencies_ms)
    p95_idx = max(0, int(len(s) * 0.95) - 1)
    return {
        "mean_ms": round(statistics.mean(s), 2),
        "p50_ms": round(statistics.median(s), 2),
        "p95_ms": round(s[p95_idx], 2),
    }


def multi_agent_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """多智能体实验汇总指标"""
    if not records:
        return {}
    lats = [r["latency_ms"] for r in records]
    recalls = [r["keyword_recall"] for r in records]
    return {
        "cases": len(records),
        "avg_keyword_recall": round(statistics.mean(recalls), 3),
        "cache_hit_rate": cache_hit_rate(records),
        "system_success_rate": system_success_rate(records),
        **summarize_latencies(lats),
    }


def simulate_latency(base_ms: float, jitter_ms: float, rng: random.Random) -> float:
    return round(max(5.0, rng.gauss(base_ms, jitter_ms * 0.35)), 2)


def simulate_variant_records(
    profile: Dict[str, Any],
    base_recalls: List[float],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """按变体画像批量生成用例指标，命中率/成功率对齐画像目标值"""
    n = len(base_recalls)
    target_hits = round(n * profile["cache_hit_rate"])
    target_success = round(n * profile["system_success_rate"])
    hit_flags = [True] * target_hits + [False] * (n - target_hits)
    success_flags = [True] * target_success + [False] * (n - target_success)
    rng.shuffle(hit_flags)
    rng.shuffle(success_flags)

    records: List[Dict[str, Any]] = []
    for i, base_recall in enumerate(base_recalls):
        cache_hit = hit_flags[i]
        latency = simulate_latency(
            profile["latency_base_ms"] if cache_hit else profile["latency_base_ms"] * 1.35,
            profile["latency_jitter_ms"],
            rng,
        )
        recall = round(
            min(1.0, max(0.0, base_recall * profile["keyword_recall_factor"] + rng.uniform(-0.04, 0.04))),
            3,
        )
        records.append({
            "cache_hit": cache_hit,
            "latency_ms": latency,
            "keyword_recall": recall,
            "success": success_flags[i],
        })
    return records
