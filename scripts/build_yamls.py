#!/usr/bin/env python3
"""Regenerate SRD ablation YAML configs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    path = SCRIPTS / "build_srd_ablation.py"
    print(f"\n=== {path.name} ===")
    subprocess.check_call([sys.executable, str(path)])


if __name__ == "__main__":
    main()
