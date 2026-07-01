#!/usr/bin/env python3
"""一键运行对比 + 消融实验并生成报告"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="运行全部实验")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "experiments", "results", f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    py = sys.executable
    if not args.skip_comparison:
        subprocess.check_call([py, os.path.join(ROOT, "experiments", "run_comparison.py"), "--out", out_dir])
    if not args.skip_ablation:
        subprocess.check_call([py, os.path.join(ROOT, "experiments", "run_ablation.py"), "--out", out_dir])

    subprocess.check_call([py, os.path.join(ROOT, "experiments", "report.py"), out_dir])
    print(f"\n=== 全部实验完成 ===\n结果与报告: {out_dir}/REPORT.md")


if __name__ == "__main__":
    main()
