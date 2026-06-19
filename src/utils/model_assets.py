"""MediaPipe 模型资源管理"""

from __future__ import annotations

import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def ensure_models() -> dict:
    """确保 MediaPipe 模型文件已下载"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {}
    for name, url in MODEL_URLS.items():
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            logger.info("下载模型: %s ...", name)
            urllib.request.urlretrieve(url, path)
            logger.info("已下载: %s", path)
        paths[name] = path
    return paths
