#!/bin/bash
# 一键运行对比与消融实验
cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || true
python experiments/run_all.py "$@"
