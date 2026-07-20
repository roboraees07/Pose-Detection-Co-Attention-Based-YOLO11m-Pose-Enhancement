#!/usr/bin/env python3
"""Confidence ablation for YOLO11m-Pose (aug ON and OFF checkpoints)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdca_yolo11.constants import CONF_SWEEP_VALUES  # noqa: E402
from pdca_yolo11.eval.engine import evaluate  # noqa: E402
from pdca_yolo11.eval.export import paper_row, write_csv  # noqa: E402
from pdca_yolo11.experiments import CONF_SWEEP_MODELS  # noqa: E402
from pdca_yolo11.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402
from pdca_yolo11.train_utils import register_modules  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO11m-Pose confidence ablation study")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, nargs="*", default=list(CONF_SWEEP_VALUES))
    args = p.parse_args()

    weed_rows, overall_rows, json_out = [], [], {"weed": [], "overall": []}

    for exp in CONF_SWEEP_MODELS:
        register_modules(exp)
        weights = RUNS_DIR / exp.run_name / "weights" / "best.pt"
        if not weights.is_file():
            print(f"SKIP {exp.id}: missing {weights}")
            continue
        aug_label = "With augmentation" if exp.dat else "No augmentation"
        run_dir = RUNS_DIR / exp.run_name

        for conf in args.conf:
            print(f"\n{exp.display_name} @ conf={conf}")
            res = evaluate(weights, conf_thr=conf, device=args.device, run_dir=run_dir)
            base_vs = "— (reference)" if exp.dat else "—"
            for scope, key, rows in (
                ("Weed (class 8)", "weed", weed_rows),
                ("Overall (9-class macro)", "overall", overall_rows),
            ):
                m = res[f"{key}_metrics"]
                tr = res[f"{key}_table_row"]
                row = paper_row(
                    exp,
                    scope,
                    {"epoch_stops": res["epoch_stops"]},
                    tr,
                    res["model_info"],
                    vs_baseline=base_vs if exp.dat else "—",
                    augmentation=aug_label,
                    confidence=conf,
                )
                row["Confidence"] = conf
                rows.append(row)
                json_out[key].append(row)

    out_dir = RESULTS_DIR / "conf_sweep"
    write_csv(weed_rows, out_dir / "conf_sweep_weed.csv")
    write_csv(overall_rows, out_dir / "conf_sweep_overall.csv")
    (out_dir / "conf_sweep.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir}/conf_sweep_weed.csv")
    print(f"Wrote {out_dir}/conf_sweep_overall.csv")


if __name__ == "__main__":
    main()
