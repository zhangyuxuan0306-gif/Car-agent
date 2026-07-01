"""以点击坐标为中心切出 3 个局部街景块"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.agents.schemas import ImageTile


def slice_tiles(
    frame: np.ndarray,
    click_x: int,
    click_y: int,
    tile_size_ratio: float = 0.35,
    offsets: Tuple[float, ...] = (-0.18, 0.0, 0.18),
) -> List[Tuple[ImageTile, np.ndarray]]:
    """
    以 (click_x, click_y) 为中心，按水平偏移切出 A/B/C 三块图像。
    返回 (ImageTile 元数据, 裁剪图像) 列表。
    """
    h, w = frame.shape[:2]
    tw = max(64, int(w * tile_size_ratio))
    th = max(64, int(h * tile_size_ratio))

    tiles: List[Tuple[ImageTile, np.ndarray]] = []
    labels = ("A", "B", "C")

    for label, off in zip(labels, offsets):
        cx = int(np.clip(click_x + off * w, tw // 2, w - tw // 2 - 1))
        cy = int(np.clip(click_y, th // 2, h - th // 2 - 1))
        x1 = cx - tw // 2
        y1 = cy - th // 2
        x2 = x1 + tw
        y2 = y1 + th
        crop = frame[y1:y2, x1:x2].copy()
        meta = ImageTile(tile_id=label, bbox=(x1, y1, x2, y2), center=(cx, cy))
        tiles.append((meta, crop))

    return tiles
