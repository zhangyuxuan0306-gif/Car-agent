# 智能座舱后台静默感知多智能体系统

基于自训练 YOLO 识别北京 CBD 建筑，采用 **Vision ×3 并行 + RAG + Writer + Critic** 多智能体协作架构。系统支持后台静默感知与点击瞬时唤醒，问答默认纯本地（不联网、毫秒级）。

## 快速开始

```bash
source venv/bin/activate

# 多智能体可交互视频（推荐）
python main.py --mode agent --video data/videos/GuoMao.mp4

# 单 Agent 视频演示
python main.py --mode video

# 定量评估实验（对比 + 消融 + 中文报告）
python main.py --mode exp
# 或
python experiments/run_all.py

# Web 界面
python main.py --mode ui
```

## 多智能体架构

| 阶段 | Agent | 职责 |
|------|-------|------|
| 后台 | Vision ×3 | 左/中/右三区静默并发感知，写入 PerceptionCache |
| 点击 | Master | 缓存命中瞬时唤醒，或 fallback 串行识别 |
| 协作 | RAG Agent | 知识库检索，组装结构化上下文 |
| 协作 | Writer Agent | 生成导游播报词 |
| 质检 | Critic Agent | 对抗审查，不合格则退回重写 |

独立 TUI 演示（asyncio + Rich 看板）：

```bash
python smart_cockpit_agents.py
```

## 项目结构

```
car-agent/
├── config.yaml
├── main.py
├── smart_cockpit_agents.py    # 多智能体 TUI 演示
├── models/
│   ├── yolo/                  # 自训练 YOLO 推理
│   ├── face_landmarker.task
│   └── hand_landmarker.task
├── data/
│   ├── videos/GuoMao.mp4
│   └── knowledge/buildings.json
├── experiments/               # 多智能体定量评估（见 experiments/README.md）
│   ├── run_all.py
│   └── benchmarks/qa.json
└── src/
    ├── agents/                # Master / Vision / RAG / Writer / Critic
    ├── video/cockpit_agent_video.py
    ├── perception/custom_yolo_detector.py
    └── multimodal/local_qa_engine.py
```

## 实验评估

运行 `python experiments/run_all.py` 后，在 `experiments/results/run_*/` 生成：

- `comparison_multi_agent.json` — 多智能体 vs 基线方案对比
- `ablation_multi_agent.json` — A0~A3 消融实验
- `REPORT.md` — 全中文报告（含 Rich 表格，可直接引用至技术报告）

详见 [experiments/README.md](experiments/README.md)。

## License

MIT
