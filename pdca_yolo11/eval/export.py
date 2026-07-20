"""Export evaluation results to paper-style CSV tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from pdca_yolo11.constants import BATCH, EPOCHS, IMGSZ, LR_PROTOCOL, PATIENCE
from pdca_yolo11.experiments import Experiment, module_flags_row, srd_flags_row, tick


def paper_row(
    exp: Experiment,
    scope: str,
    metrics: dict,
    table_row: dict,
    model_info: dict,
    *,
    vs_baseline: str = "—",
    augmentation: str | None = None,
    confidence: float | None = None,
) -> dict:
    aug = augmentation if augmentation is not None else ("ON" if exp.dat else "OFF")
    row = {
        "Model": exp.display_name,
        "Method": exp.display_name,
        "Augmentation": aug,
        "Dat": tick(exp.dat),
        "Scope": scope,
        "mAP_kpt/%": table_row["Pose mAP50(%)"],
        "F1/%": table_row["Pose F1(%)"],
        "Params/M": model_info.get("Params/M", ""),
        "FPS": model_info.get("FPS", ""),
        "vs YOLO11m": vs_baseline,
        "Epoch stops": metrics.get("epoch_stops", "?"),
        "Det P(%)": table_row["Det P(%)"],
        "Det R(%)": table_row["Det R(%)"],
        "Det F1(%)": table_row["Det F1(%)"],
        "Det mAP50(%)": table_row["Det mAP50(%)"],
        "Det mAP50-95(%)": table_row["Det mAP50-95(%)"],
        "Pose P(%)": table_row["Pose P(%)"],
        "Pose R(%)": table_row["Pose R(%)"],
        "Pose F1(%)": table_row["Pose F1(%)"],
        "Pose mAP50(%)": table_row["Pose mAP50(%)"],
        "Pose mAP50-95(%)": table_row["Pose mAP50-95(%)"],
        "MEA(px)": table_row["MEA(px)"],
        "CAR(%)": table_row["CAR(%)"],
        "RMSE(%)": table_row["RMSE(%)"],
        "time(ms/img)": table_row["mean time(ms/img)"],
        "Parameters": model_info.get("Parameters", ""),
        "Layers": model_info.get("Layers", ""),
        "GFLOPs": model_info.get("GFLOPs", ""),
        "Test Dataset": "CropsOrWeed9 (test)",
        "Image Size": IMGSZ,
        "Batch (req)": BATCH,
        "Batch (used)": BATCH,
        "Epochs (max)": EPOCHS,
        "Learning rate": LR_PROTOCOL,
        "Patience": PATIENCE,
        "Optimizer": "SGD",
        "Run": exp.run_name,
    }
    row.update(module_flags_row(exp))
    row.update(srd_flags_row(exp))
    if confidence is not None:
        row["Confidence"] = confidence
    return row


def write_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_eval_bundle(eval_json: Path) -> dict:
    return json.loads(eval_json.read_text(encoding="utf-8"))
