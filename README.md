# 智能座舱后台静默感知多智能体系统

为了更加紧密地契合“智能座舱车、人、路、景多模态协同”的主题，并在工程落地中体现最前沿的多智能体（Multi-Agent）协同与 OpenCode 工具链（如本地/沙盒代码执行环境）的架构，我们可以把 README 的核心功能重构为基于 Multi-Agent 架构的分布式感知与决策系统。

🚀 核心功能与多智能体协作机制 (Core Features & Multi-Agent Collaboration)

本系统打破了传统单体系统的感知瓶颈，采用基于大模型的多智能体（Multi-Agent）协同架构。通过引入 OpenCode 智能代码执行环境，各 Agent 能够动态生成并运行代码，打通了舱内“人眼/手势”行为感知与舱外“实时街景”目标检测的物理边界，实现了真正意义上的“人-车-路-景”多模态智能协同交互。

                       ┌────────────────────────┐
                       │   用户语音/行为输入     │
                       └───────────┬────────────┘
                                   ▼
                       ┌────────────────────────┐
                       │  Agent 1: 视线决策官   │ ──> 捕捉红点与 YOLO 框空间匹配
                       └───────────┬────────────┘
                                   ▼
                       ┌────────────────────────┐
                       │  Agent 2: 动态执行官   │ ──> 调用 OpenCode 沙盒执行时空校准与清洗代码
                       └───────────┬────────────┘
                                   ▼
                       ┌────────────────────────┐
                       │  Agent 3: 知识检索官   │ ──> 两阶段混合检索 (Local JSON / Web RAG)
                       └────────────────────────┘

1. 舱外街景感知与跨模态空间协同 (Agent 1: 视线与空间决策官)

    高精度车路协同感知：系统基于自训练的 YOLO 目标检测模型，针对城市典型 CBD 区域（如国贸核心区）的 7 类主要建筑目标进行高精度、低延迟的实时多目标跟踪与识别（路/景感知）。

    座舱交互意图捕捉（红点落点机制）：乘客可通过 Web UI 界面直接点击选中视线所及的建筑目标。系统精准捕捉点击位置并生成交互“红点”（像素焦点坐标）。

        💡 前瞻性设计：当前的“鼠标点击红点”在未来可无缝等效替代为舱内硬件的视线追踪（Gaze Tracking）注视区域或手势指引（Gesture）的动态投射点（人感知）。

    多模态实体空间对齐：空间决策 Agent 实时订阅 YOLO 检测框（Bounding Box）流与交互红点坐标，通过动态空间求交算法，在毫秒级内精准锁定乘客当前的交互意图。

2. OpenCode 智能代码执行与数据清洗 (Agent 2: 动态执行官)

为了应对复杂的车载传感器数据异构性与复杂的空间几何计算，系统集成了 OpenCode 工具链：

    动态几何计算沙盒：当传感器回传的视线红点与建筑框存在高速运动带来的时空偏差时，动态执行官 Agent 会根据当前车速、视线延迟等参数，自动编写 Python 空间校准代码，并在 OpenCode 安全沙盒环境中实时编译运行，实现自适应的像素级动态补偿。

    结构化数据按需抽取：在处理联网检索回来的海量非结构化网页文本时，Agent 通过 OpenCode 动态生成正则表达式或轻量化数据清洗脚本，在本地执行环境中对文本进行剪裁与降噪，将非结构化文本秒级转化为标准输入流。

3. 两阶段混合检索问答引擎 (Agent 3: 知识检索官)

当用户针对窗外街景实体进行提问时，知识检索 Agent 具备极强的长尾问题处理能力与感知响应健壮性，采用以下两阶段智能兜底策略：

    第一阶段：本地精准匹配 (Local Knowledge Retrieval)
    优先检索车载本地的结构化知识库（如 buildings.json）。若成功命中，则直接提取相关的地理位置、建筑背景、周边评测等高价值数据，由 Agent 快速组织语言进行高时效的局部回复。

    第二阶段：联网动态检索与 Web RAG (Web-Driven RAG Backup)
    若用户的提问涉及冷门场景、突发路况或超出了本地库的知识边界，Agent 将自动触发联网搜索机制。系统利用搜索引擎 API 针对当前识别到的实体动态抓取多源网页的实时信息。

    多Agent协同润色：抓取到的碎片化文本经过 OpenCode 清洗后，由大语言模型利用强大的文本压缩与信息抽取能力，去除广告和网页噪声进行逻辑重组与润色，最终为用户平滑输出结构清晰、切中要害的高质量回答。

🛠️ 技术栈 (Tech Stack)

    感知层：Python, PyTorch, YOLOv8 (建筑目标检测与跟踪)

    智能体框架：Multi-Agent Collaboration Framework (Agent 间异步通信与状态机管理)

    执行工具链：OpenCode Runtime Environment (代码生成与沙盒本地执行)

    检索生成：LangChain / LlamaIndex, Web Search API, 大语言模型 (LLM/VLM)
    
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
