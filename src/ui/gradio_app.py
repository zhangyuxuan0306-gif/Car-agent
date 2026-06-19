"""Gradio 交互界面"""

from __future__ import annotations

import logging
import os
import sys

import cv2
import gradio as gr
import numpy as np

# 确保项目根目录在 path 中
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import CockpitScenePipeline

logger = logging.getLogger(__name__)

_pipeline: CockpitScenePipeline = None


def get_pipeline() -> CockpitScenePipeline:
    global _pipeline
    if _pipeline is None:
        config_path = os.path.join(ROOT, "config.yaml")
        _pipeline = CockpitScenePipeline(config_path)
    return _pipeline


def process_image(image: np.ndarray, click_x: float = 0.5, click_y: float = 0.5):
    """处理上传的街景图片/视频帧，click 坐标用于模拟注视点"""
    if image is None:
        return None, "请上传街景图片或视频", "", "", ""

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    point_px = (int(click_x * w), int(click_y * h))

    pipeline = get_pipeline()
    state = pipeline.process_frame(frame, point_px=point_px)

    vis = cv2.cvtColor(state.visualization, cv2.COLOR_BGR2RGB)

    det_info = f"检测到 {len(state.detections)} 个建筑目标（上限4个）"
    if state.detections:
        det_info += "\n" + "\n".join(
            f"  • [{i}] {d.label} ({d.confidence:.2f})"
            for i, d in enumerate(state.detections[:4])
        )

    gaze_info = "未检测到人脸"
    if state.gaze and state.gaze.face_detected:
        gaze_info = (
            f"注视点: ({state.gaze.gaze_point[0]:.2f}, {state.gaze.gaze_point[1]:.2f})\n"
            f"置信度: {state.gaze.confidence:.2f}\n"
            f"头部姿态: pitch={state.gaze.head_pose[0]:.1f}° "
            f"yaw={state.gaze.head_pose[1]:.1f}°"
        )

    gesture_info = "未检测到手势"
    if state.gesture and state.gesture.hands:
        gesture_info = f"主手势: {state.gesture.primary_gesture.value}\n"
        gesture_info += f"检测到 {len(state.gesture.hands)} 只手"

    att_info = "暂无关注目标"
    if state.attention and state.attention.best_target:
        bt = state.attention.best_target
        att_info = (
            f"关注建筑: {bt.detection.label}\n"
            f"综合注意力得分: {bt.total_score:.3f}\n"
            f"  眼动得分: {bt.gaze_score:.3f}\n"
            f"  手势得分: {bt.gesture_score:.3f}\n"
            f"  检测得分: {bt.detection_score:.3f}\n"
            f"  多帧确认: {'是' if bt.is_confirmed else '否'}"
        )

    status = (
        f"处理耗时: {state.processing_time_ms:.1f}ms\n\n"
        f"【视觉检测】\n{det_info}\n\n"
        f"【眼动追踪】\n{gaze_info}\n\n"
        f"【手势识别】\n{gesture_info}\n\n"
        f"【注意力融合】\n{att_info}"
    )

    return vis, status, "", "", ""


def analyze_building(image: np.ndarray):
    """分析关注建筑并生成介绍"""
    if image is None:
        return "请先上传并处理街景图片", "", None

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    pipeline = get_pipeline()

    # 如果还没处理过，先处理
    if pipeline.state.frame is None:
        pipeline.process_frame(frame)

    identification, description = pipeline.analyze_target(frame)

    crop_display = None
    if pipeline.state.target_crop is not None:
        crop_display = cv2.cvtColor(pipeline.state.target_crop, cv2.COLOR_BGR2RGB)

    kb = pipeline.state.knowledge_context
    knowledge_text = kb if kb else "（未匹配到知识库条目，已由大模型独立推理）"

    return identification, description, knowledge_text, crop_display


def chat(question: str, history: list, progress=gr.Progress()):
    if not question.strip():
        return history, ""

    pipeline = get_pipeline()
    progress(0.2, desc="检索本地知识库")
    answer = pipeline.ask(question)
    progress(1.0, desc="完成")
    history = history + [(question, answer)]
    return history, ""


def load_vlm_model():
    """初始化知识服务"""
    pipeline = get_pipeline()
    return pipeline.load_vlm()


def create_app() -> gr.Blocks:
    """创建 Gradio 应用"""

    with gr.Blocks(
        title="智能座舱街景理解系统",
    ) as app:
        gr.Markdown(
            "# 🏙️ 智能座舱街景理解系统\n"
            "融合 **视觉检测 · 眼动追踪 · 手势识别 · 多模态大模型**，"
            "推断乘客关注的建筑目标并生成专业化语义介绍。",
            elem_classes=["main-title"],
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📷 输入（图片 / 视频帧）")
                input_image = gr.Image(label="街景图片或视频截图", type="numpy")
                with gr.Row():
                    click_x = gr.Slider(0, 1, value=0.5, label="注视点 X（模拟鼠标点击）")
                    click_y = gr.Slider(0, 1, value=0.4, label="注视点 Y")
                with gr.Row():
                    btn_process = gr.Button("🔍 多模态感知分析", variant="primary")
                    btn_load_vlm = gr.Button("🧠 加载问答大模型")
                vlm_status = gr.Textbox(label="大模型状态", value="点击加载问答大模型（首次约半分钟）", interactive=False)
                gr.Markdown(
                    "> **视频模式（推荐）**: `python main.py --mode video` — "
                    "播放窗外景，点击建筑识别；**终端输入问题并回车**即可问答"
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🎯 感知结果")
                output_vis = gr.Image(label="多模态融合可视化")
                status_text = gr.Textbox(label="感知状态", lines=12, elem_classes=["status-box"])

        gr.Markdown("---")
        gr.Markdown("### 🏛️ 建筑目标分析与语义生成")

        with gr.Row():
            btn_analyze = gr.Button("🏗️ 分析关注建筑并生成介绍", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                target_crop = gr.Image(label="关注目标裁剪")
            with gr.Column():
                identification = gr.Textbox(label="建筑识别结果", lines=6)
                description = gr.Textbox(label="专业化语义介绍", lines=10)
                knowledge = gr.Textbox(label="知识库补全", lines=4)

        gr.Markdown("---")
        gr.Markdown("### 💬 交互问答")

        chatbot = gr.Chatbot(label="建筑问答", height=400)
        with gr.Row():
            question_input = gr.Textbox(
                label="您的问题",
                placeholder="例如：这栋建筑是什么风格？建于哪一年？有什么历史故事？",
                scale=4,
            )
            btn_ask = gr.Button("发送", variant="primary", scale=1)

        # 事件绑定
        btn_process.click(
            process_image,
            inputs=[input_image, click_x, click_y],
            outputs=[output_vis, status_text, identification, description, knowledge],
        )
        btn_load_vlm.click(load_vlm_model, outputs=[vlm_status])
        btn_analyze.click(
            analyze_building,
            inputs=[input_image],
            outputs=[identification, description, knowledge, target_crop],
        )
        btn_ask.click(chat, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
        question_input.submit(chat, inputs=[question_input, chatbot], outputs=[chatbot, question_input])

        gr.Markdown(
            "---\n"
            "**架构**: 自训练 YOLOv8 + 本地/联网资料 + Qwen2.5 整理问答（点击简介≤30字）"
        )

    return app


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    app = create_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .status-box { font-family: monospace; font-size: 0.9em; }
        """,
    )


if __name__ == "__main__":
    import yaml
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ui_cfg = cfg.get("ui", {})
    launch(
        host=ui_cfg.get("host", "0.0.0.0"),
        port=ui_cfg.get("port", 7860),
        share=ui_cfg.get("share", False),
    )
