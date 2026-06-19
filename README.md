# 智能座舱街景理解系统

基于自训练 YOLO 识别北京 CBD 7 类建筑。**问答默认纯本地**（不联网、毫秒级），画面底部显示进度条。

## 快速开始

```bash
source venv/bin/activate
python main.py --mode video          # 视频点击识别（推荐）
python main.py --mode ui             # Web 界面
```

## 项目结构

```
car/
├── config.yaml
├── main.py
├── models/
│   ├── yolo/                  # 自训练 YOLO 推理（权重+代码）
│   ├── face_landmarker.task   # MediaPipe 眼动
│   └── hand_landmarker.task   # MediaPipe 手势
├── data/
│   ├── videos/GuoMao.mp4
│   └── knowledge/buildings.json
└── src/
    ├── video/cockpit_video.py
    ├── perception/custom_yolo_detector.py
    └── multimodal/local_qa_engine.py
```

## License

MIT
