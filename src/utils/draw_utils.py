"""中文文字绘制工具（OpenCV 不支持中文）"""

from __future__ import annotations

import os
import textwrap
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_FONT_CACHE = {}


def _get_font(size: int = 22) -> ImageFont.FreeTypeFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for fp in [
        os.path.join(ROOT, "models", "yolo", "model_data", "simhei.ttf"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]:
        if os.path.isfile(fp):
            _FONT_CACHE[size] = ImageFont.truetype(fp, size)
            return _FONT_CACHE[size]
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_text_cn(
    frame: np.ndarray,
    text: str,
    anchor: Tuple[int, int],
    font_size: int = 22,
    fill: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (20, 20, 20),
    max_width: int = 22,
) -> np.ndarray:
    """在 BGR 图像上绘制中文文字"""
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _get_font(font_size)
    x, y = anchor
    lines = textwrap.fill(text, width=max_width).split("\n")
    max_w = max((draw.textlength(ln, font=font) for ln in lines), default=0)
    draw.rectangle(
        [x - 4, y - 4, x + int(max_w) + 10, y + len(lines) * (font_size + 4) + 6],
        fill=bg,
    )
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font_size + 4
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_bottom_banner(
    frame: np.ndarray,
    text: str,
    font_size: int = 18,
    max_width: int = 46,
    lines_max: int = 4,
) -> np.ndarray:
    """视频底部半透明字幕条 — 展示联网/大模型/导游词全文"""
    if not text or not text.strip():
        return frame
    h, w = frame.shape[:2]
    wrapped = textwrap.fill(text.strip(), width=max_width)
    lines = wrapped.split("\n")[:lines_max]
    if len(wrapped.split("\n")) > lines_max:
        lines[-1] = lines[-1][: max_width - 1] + "…"

    bar_h = len(lines) * (font_size + 6) + 16
    y0 = h - bar_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    y = y0 + 8
    for line in lines:
        frame = draw_text_cn(frame, line, (12, y), font_size=font_size, max_width=max_width + 4, bg=(15, 15, 15))
        y += font_size + 6
    return frame


def draw_progress_bar(
    frame: np.ndarray,
    progress: float,
    label: str = "处理中",
) -> np.ndarray:
    h, w = frame.shape[:2]
    progress = max(0.0, min(1.0, progress))
    bar_w, bar_h = min(420, w - 80), 20
    x0 = (w - bar_w) // 2
    y0 = h - 56
    pct = int(progress * 100)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 2, y0 - 28), (x0 + bar_w + 2, y0 + bar_h + 8), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (80, 80, 80), -1)
    fill = max(0, int(bar_w * progress))
    if fill > 0:
        cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bar_h), (0, 200, 90), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)

    tip = label or "处理中"
    frame = draw_text_cn(frame, f"{tip}  {pct}%", (x0, y0 - 30), font_size=18, max_width=30)
    return frame
