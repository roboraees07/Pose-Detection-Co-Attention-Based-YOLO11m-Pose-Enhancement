#!/usr/bin/env python3
"""List all registered experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdca_yolo11.experiments import GROUPS, module_flags_row, srd_flags_row, tick  # noqa: E402


def main() -> None:
    for group, exps in GROUPS.items():
        print(f"\n{'='*60}\n{group} ({len(exps)} experiments)\n{'='*60}")
        for exp in exps:
            flags = module_flags_row(exp) if group != "srd_ablation" else srd_flags_row(exp)
            flag_str = " ".join(f"{k}={v}" for k, v in flags.items())
            print(f"  {exp.id:30s}  {exp.display_name}")
            print(f"    run: {exp.run_name}")
            print(f"    model: {exp.model}")
            if flag_str.strip():
                print(f"    {flag_str}")


if __name__ == "__main__":
    main()
