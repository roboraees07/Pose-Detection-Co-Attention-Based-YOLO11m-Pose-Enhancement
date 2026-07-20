#!/usr/bin/env python3
"""Export paper tables from evaluation JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdca_yolo11.constants import EVAL_CONF  # noqa: E402
from pdca_yolo11.eval.export import paper_row, write_csv  # noqa: E402
from pdca_yolo11.experiments import GROUPS  # noqa: E402
from pdca_yolo11.paths import RESULTS_DIR, RUNS_DIR  # noqa: E402


def load_eval(exp, conf: float) -> dict | None:
    path = RUNS_DIR / f"{exp.run_name}_eval_conf{str(conf).replace('.', '')}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def export_group(group_name: str, conf: float = EVAL_CONF) -> None:
    exps = GROUPS[group_name]
    weed_rows, overall_rows = [], []
    baseline_weed_map = None

    # Find YOLO11m aug baseline for vs column
    for exp in exps:
        if exp.id in ("ladder_baseline", "method_yolo11m_baseline"):
            ev = load_eval(exp, conf)
            if ev:
                baseline_weed_map = ev["weed_table_row"]["Pose mAP50(%)"]

    seen = set()
    for exp in exps:
        if exp.run_name in seen:
            continue
        seen.add(exp.run_name)
        ev = load_eval(exp, conf)
        if not ev:
            print(f"SKIP {exp.id}: no eval JSON")
            continue
        for scope, key, rows in (
            ("Weed (class 8)", "weed", weed_rows),
            ("Overall (9-class macro)", "overall", overall_rows),
        ):
            vs = "— (reference)" if exp.id in ("ladder_baseline", "method_yolo11m_baseline") else "—"
            if baseline_weed_map is not None and key == "weed" and exp.id not in ("ladder_baseline", "method_yolo11m_baseline"):
                delta = round(ev["weed_table_row"]["Pose mAP50(%)"] - baseline_weed_map, 2)
                vs = f"{delta:+.2f}"
            row = paper_row(
                exp,
                scope,
                {"epoch_stops": ev.get("epoch_stops", "?")},
                ev[f"{key}_table_row"],
                ev["model_info"],
                vs_baseline=vs,
            )
            rows.append(row)

    out = RESULTS_DIR / group_name
    write_csv(weed_rows, out / f"{group_name}_weed.csv")
    write_csv(overall_rows, out / f"{group_name}_overall.csv")
    print(f"Wrote {out}/")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(GROUPS))
    p.add_argument("--conf", type=float, default=EVAL_CONF)
    args = p.parse_args()
    export_group(args.group, args.conf)


if __name__ == "__main__":
    main()
