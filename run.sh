#!/bin/bash
# 智能座舱街景理解系统 - 启动脚本

set -e
cd "$(dirname "$0")"

VENV_DIR="venv"

# 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo ">>> 创建 Python 虚拟环境 ..."
    python3.10 -m pip install --user virtualenv -q 2>/dev/null || true
    python3.10 -m virtualenv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 安装依赖
if [ ! -f "$VENV_DIR/.installed" ]; then
    echo ">>> 安装依赖（首次运行，可能需要较长时间）..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    touch "$VENV_DIR/.installed"
fi

echo ">>> 启动智能座舱街景理解系统 ..."
echo "    Web界面: python main.py --mode ui"
echo "    视频演示: python main.py --mode video"
python main.py --mode ui --host 0.0.0.0 --port 7860 "$@"
