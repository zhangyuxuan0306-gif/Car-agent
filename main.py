#!/usr/bin/env python3
"""智能座舱街景理解系统 - 主入口"""

import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="智能座舱街景理解系统")
    parser.add_argument("--mode", choices=["ui", "demo", "image", "video", "agent", "exp"], default="ui",
                        help="运行模式: agent=多智能体可点击视频, ui=Web界面, video=单Agent视频, demo=命令行演示, image=单图, exp=实验")
    parser.add_argument("--video", type=str, default=None, help="窗外景视频路径")
    parser.add_argument("--image", type=str, default=None, help="输入图片路径")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gradio 公网分享")
    args = parser.parse_args()

    if args.mode == "ui":
        from src.ui.gradio_app import launch
        logger.info("启动 Web 界面 ...")
        launch(host=args.host, port=args.port, share=args.share)

    elif args.mode == "demo":
        run_demo()

    elif args.mode == "image":
        if not args.image:
            logger.error("请通过 --image 指定图片路径")
            sys.exit(1)
        run_image_analysis(args.image)

    elif args.mode == "video":
        run_video_demo(args.video)

    elif args.mode == "agent":
        run_agent_demo(args.image, args.video)

    elif args.mode == "exp":
        from experiments.run_all import main as run_experiments
        run_experiments()


def run_demo():
    """命令行演示模式"""
    import cv2
    from src.pipeline import CockpitScenePipeline

    demo_dir = os.path.join(ROOT, "data", "demo")
    os.makedirs(demo_dir, exist_ok=True)

    # 查找演示图片
    demo_images = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not demo_images:
        logger.info("未找到演示图片，正在下载示例街景图 ...")
        download_demo_image(demo_dir)
        demo_images = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    img_path = os.path.join(demo_dir, demo_images[0])
    logger.info("使用演示图片: %s", img_path)

    pipeline = CockpitScenePipeline(os.path.join(ROOT, "config.yaml"))

    frame = cv2.imread(img_path)
    if frame is None:
        logger.error("无法读取图片: %s", img_path)
        sys.exit(1)

    logger.info("=== 阶段1: 多模态感知 ===")
    state = pipeline.process_frame(frame)
    logger.info("检测到 %d 个建筑目标", len(state.detections))
    for d in state.detections:
        logger.info("  - %s (%.2f)", d.label, d.confidence)

    if state.attention and state.attention.best_target:
        bt = state.attention.best_target
        logger.info("关注目标: %s (得分 %.3f)", bt.detection.label, bt.total_score)

    out_vis = os.path.join(demo_dir, "output_visualization.jpg")
    cv2.imwrite(out_vis, state.visualization)
    logger.info("可视化结果已保存: %s", out_vis)

    logger.info("=== 阶段2: 建筑简介（本地优先+联网） ===")
    identification, description = pipeline.analyze_target(frame)
    logger.info("\n【建筑识别】%s", identification)
    logger.info("【简介≤30字】%s", description)

    logger.info("=== 阶段3: 交互问答 ===")
    answer = pipeline.ask("这栋建筑有什么历史文化价值？")
    logger.info("\n【问答】\n%s", answer)

    pipeline.cleanup()
    logger.info("演示完成!")


def run_video_demo(video_path: str = None):
    """车窗外景视频演示（Lenovo 原项目交互方式）"""
    import yaml
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if video_path:
        config.setdefault("video", {})["path"] = video_path
    from src.video.cockpit_video import CockpitVideoDemo
    CockpitVideoDemo(config).run()


def run_image_analysis(image_path: str):
    """单图分析模式"""
    import cv2
    from src.pipeline import CockpitScenePipeline

    pipeline = CockpitScenePipeline(os.path.join(ROOT, "config.yaml"))
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("无法读取: %s", image_path)
        sys.exit(1)

    state = pipeline.process_frame(frame)
    print(f"\n检测到 {len(state.detections)} 个目标")
    identification, description = pipeline.analyze_target(frame)
    print(f"\n=== 识别 ===\n{identification}")
    print(f"\n=== 介绍 ===\n{description}")
    pipeline.cleanup()


def run_agent_demo(image_path: str = None, video_path: str = None):
    """多智能体协作演示 — 默认可交互视频，--image 时单帧分析"""
    import yaml

    if image_path and not video_path:
        _run_agent_single_image(image_path)
        return

    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if video_path:
        config.setdefault("video", {})["path"] = video_path

    from src.video.cockpit_agent_video import CockpitAgentVideoDemo
    CockpitAgentVideoDemo(config).run()


def _run_agent_single_image(image_path: str):
    """单图多智能体分析（非交互）"""
    import cv2
    from src.agents.master import MultiAgentOrchestrator

    config_path = os.path.join(ROOT, "config.yaml")
    orchestrator = MultiAgentOrchestrator(
        config_path,
        on_dashboard_update=lambda text: print(f"\n--- 看板 ---\n{text}\n"),
    )

    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("无法读取: %s", image_path)
        sys.exit(1)

    h, w = frame.shape[:2]
    click_x, click_y = w // 2, h // 2

    logger.info("=== 多智能体协作（单图）===")
    output, vis = orchestrator.run(frame, click_x, click_y)

    print("\n" + "=" * 60)
    print("【最终导游播报词】")
    print(output.script)
    print("\n【识别建筑】", "、".join(output.buildings))
    print("=" * 60)

    out_dir = os.path.join(ROOT, "data", "demo")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "agent_output.jpg")
    cv2.imwrite(out_path, vis)
    logger.info("可视化结果: %s", out_path)


def download_demo_image(demo_dir: str):
    """下载示例街景图片"""
    import urllib.request
    urls = [
        "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=1280",  # 城市天际线
    ]
    for i, url in enumerate(urls):
        path = os.path.join(demo_dir, f"demo_cityscape_{i}.jpg")
        try:
            urllib.request.urlretrieve(url, path)
            logger.info("已下载: %s", path)
            return
        except Exception as e:
            logger.warning("下载失败 %s: %s", url, e)

    # 生成一张简单的合成图
    import numpy as np
    import cv2
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (180, 200, 220)  # 天空
    cv2.rectangle(img, (100, 300), (350, 700), (80, 80, 100), -1)
    cv2.rectangle(img, (400, 200), (600, 700), (60, 70, 90), -1)
    cv2.rectangle(img, (650, 250), (900, 700), (90, 85, 80), -1)
    cv2.rectangle(img, (950, 150), (1200, 700), (70, 75, 85), -1)
    path = os.path.join(demo_dir, "demo_synthetic.jpg")
    cv2.imwrite(path, img)
    logger.info("已生成合成演示图: %s", path)


if __name__ == "__main__":
    main()
