#!/usr/bin/env python3
"""Evaluate trained experiment(s) on CropsOrWeed9 test set (weed + overall @ conf=0.30)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdca_yolo11.constants import EVAL_CONF  # noqa: E402
from pdca_yolo11.eval.engine import evaluate, save_eval_json  # noqa: E402
from pdca_yolo11.experiments import ALL_EXPERIMENTS, GROUPS  # noqa: E402
from pdca_yolo11.paths import RUNS_DIR  # noqa: E402
from pdca_yolo11.train_utils import register_modules  # noqa: E402


def eval_one(exp_id: str, device: str, conf: float) -> None:
    exp = ALL_EXPERIMENTS[exp_id]
    register_modules(exp)
    run_dir = RUNS_DIR / exp.run_name
    weights = run_dir / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {weights} — train first with scripts/train_experiment.py --id {exp_id}")

    result = evaluate(weights, conf_thr=conf, device=device, run_dir=run_dir)
    result["experiment_id"] = exp.id
    result["display_name"] = exp.display_name
    result["run_name"] = exp.run_name

    out = RUNS_DIR / f"{exp.run_name}_eval_conf{str(conf).replace('.', '')}.json"
    save_eval_json(result, out)
    print(f"Weed  pose mAP50: {result['weed_table_row']['Pose mAP50(%)']}%")
    print(f"Overall pose mAP50: {result['overall_table_row']['Pose mAP50(%)']}%")
    print(f"Saved: {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=list(GROUPS))
    p.add_argument("--id", action="append", dest="ids")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=EVAL_CONF)
    args = p.parse_args()

    if args.group:
        ids = [e.id for e in GROUPS[args.group]]
    elif args.ids:
        ids = args.ids
    else:
        p.error("Provide --group or --id")

    seen = set()
    for eid in ids:
        exp = ALL_EXPERIMENTS[eid]
        if exp.run_name in seen:
            continue
        seen.add(exp.run_name)
        eval_one(eid, args.device, args.conf)


if __name__ == "__main__":
    main()
