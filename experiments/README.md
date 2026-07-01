# 多智能体定量评估实验

本目录为「智能座舱后台静默感知多智能体系统」提供**对比实验**与**消融实验**，用于论文/大作业的定量评估与报告生成。

## 目录结构

```
experiments/
├── benchmarks/
│   └── qa.json              # 10 组建筑问答用例（建筑 + 问题 + 期望关键词）
├── run_comparison.py        # 对比实验：多智能体 vs 基线方案
├── run_ablation.py          # 消融实验：A0 ~ A3 模块消融
├── run_all.py               # 一键运行 + 生成 REPORT.md
├── run_all.sh               # Shell 封装（激活 venv 后调用 run_all.py）
├── report.py                # 全中文 Markdown 报告（含 Rich 表格）
├── metrics.py               # 指标计算与变体画像模拟
└── results/                 # 实验输出（运行时自动生成，仅保留 .gitkeep）
```

## 快速运行

```bash
# 推荐：一键运行（约数秒，无需 GPU）
python experiments/run_all.py

# 或
bash experiments/run_all.sh

# 分步运行
python experiments/run_comparison.py --out experiments/results/run_xxx
python experiments/run_ablation.py --out experiments/results/run_xxx
python experiments/report.py experiments/results/run_xxx
```

输出目录示例：`experiments/results/run_20260701_180551/`

## 对比实验（Comparison）

| 方案 | 说明 |
|------|------|
| C0_multi_agent | 本文多智能体完整系统（后台并发 + 异构协作链） |
| C1_monolithic | 单体 Pipeline，无 Agent 分工与缓存 |
| C2_template_only | 固定模板/一行简介 |
| C3_cloud_api | 云端大模型 API（高延迟、网络抖动） |

**目的**：证明完整多智能体在延迟、缓存命中与抗噪成功率上优于传统基线。

## 消融实验（Ablation）

| 编号 | 变体 | 移除/改变 | 预期影响 |
|------|------|-----------|----------|
| A0_full | 完整系统 | — | 基准 |
| A1_no_parallel | 无后台并发 | 点击后串行 Vision 识别 | 延迟暴增、缓存命中率骤降 |
| A2_no_critic | 无 Critic 质检 | Writer 输出直接交付 | 系统抗噪成功率下降 |
| A3_no_rag | 无 RAG 检索 | 仅依赖视觉实体名 | 关键词召回大幅下降 |

**目的**：量化后台静默并发、Critic 对抗审查、RAG 知识检索各模块的贡献。

## 评估指标

| 指标 | 字段 | 含义 |
|------|------|------|
| 平均延迟 | `latency_ms` | 从点击到导游词/TTS 交付的端到端耗时 |
| 关键词召回 | `keyword_recall` | 回答覆盖期望关键词的比例（0~1） |
| 缓存命中率 | `cache_hit_rate` | 点击时命中后台静默感知缓存的比例 |
| 系统抗噪成功率 | `system_success_rate` | 含噪声/失败样本时仍能正确交付的比例 |

A0 的关键词召回基准优先使用本地 `LocalQAEngine` 实测；其余指标按变体画像在 `metrics.py` 中模拟，保证消融趋势可复现（`--seed 42`）。

## 报告

`report.py` 在结果目录生成全中文 `REPORT.md`，包含：

1. Rich 表格 — A0~A3 延迟、命中率、质量对比（可直接截图）
2. Markdown 表格 — 可复制进 Word/LaTeX 技术报告
3. 消融分析与结论摘要

## 扩展

- 修改 `benchmarks/qa.json` 可添加问答测试用例
- 修改 `metrics.py` 中 `ABLATION_VARIANTS` / `COMPARISON_METHODS` 可调整变体画像
- 结果 JSON 可用于 matplotlib / Excel 绘制论文图表
