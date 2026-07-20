#!/usr/bin/env python3
"""Regenerate all model YAML configs (YOLO11 combos, YOLO8 combos, SRD ablation)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    for name in ("build_yolo11_combos.py", "build_yolo8_combos.py", "build_srd_ablation.py"):
        path = SCRIPTS / name
        print(f"\n=== {name} ===")
        subprocess.check_call([sys.executable, str(path)])


if __name__ == "__main__":
    main()
