#!/usr/bin/env python3
"""Train one or more registered experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdca_yolo11.experiments import ALL_EXPERIMENTS, GROUPS  # noqa: E402
from pdca_yolo11.train_utils import ensure_imports, train_experiment  # noqa: E402


def main() -> None:
    ensure_imports()
    p = argparse.ArgumentParser(description="Train PDCA-YOLO11-Pose experiments")
    p.add_argument("--group", choices=list(GROUPS), help="Train all experiments in a group")
    p.add_argument("--id", action="append", dest="ids", help="Experiment id (repeatable)")
    p.add_argument("--device", default="0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    if args.group:
        exps = GROUPS[args.group]
    elif args.ids:
        exps = [ALL_EXPERIMENTS[i] for i in args.ids]
    else:
        p.error("Provide --group or --id")

    # Deduplicate by run_name (method_comparison shares runs with ladder)
    seen = set()
    for exp in exps:
        if exp.run_name in seen:
            print(f"Skip duplicate run: {exp.id} -> {exp.run_name}")
            continue
        seen.add(exp.run_name)
        train_experiment(exp, device=args.device, resume=args.resume, batch=args.batch)


if __name__ == "__main__":
    main()
