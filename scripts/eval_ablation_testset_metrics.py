#!/usr/bin/env python3
"""Full test-set localization metrics for ablation groups — matches paper eval engine.

Uses the same matching as pdca_yolo11.eval.engine (LOC_IOU_THR=0.1, conf=0.30)
and additionally reports FP%, FN%, Prec, F1 for keypoint assignment.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, "/home/raees/TUBITAK/Attention In YoloPosev11/Codes")
os.environ.setdefault("TORCH_HOME", str(PROJECT / ".torch_cache"))

import attention_modules as _am  # noqa: F401
import cv2
import numpy as np
from ultralytics import YOLO

from pdca_yolo11.attention_modules import register_attention_modules
from pdca_yolo11.constants import EVAL_CONF, EVAL_IOU, IMGSZ, LOC_IOU_THR
from pdca_yolo11.eval.engine import (
    PredItem,
    greedy_loc_match,
    load_dataset_info,
    parse_gt,
)
from pdca_yolo11.paths import DATA_YAML
from pdca_yolo11.srd_modules import register_srd_modules

RUNS = Path("/home/raees/TUBITAK/Attention In YoloPosev11/runs")
OUT_JSON = PROJECT / "results" / "ablation_testset_loc_metrics.json"
PARTIAL = OUT_JSON.with_suffix(".json.partial")


def W(name: str) -> Path:
    return RUNS / name / "weights" / "best.pt"


GROUP_A = [
    ("YOLO11m-Pose", W("yolo11m_srd_abl_a_e150_pat10_imgsz1280"), "srd"),
    ("YOLO11m-Pose+Data Augmentation", W("yolo11m_pose_baseline_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose+Data Augmentation+ECA", W("yolo11m_pose_b2_eca_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose+Data Augmentation+ECA+CBAM", W("yolo11m_pose_combo_1_2_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose+Data Augmentation+ECA+CBAM+C2PSA",
     W("yolo11m_pose_combo_1_2_4_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose+Data Augmentation+ECA+CBAM+C2PSA+PDCA",
     W("yolo11m_pose_combo_1_2_4_pdca_e150_pat10_imgsz1280"), "attn"),
    # Exact ECA+CBAM+C2PSA+SEAM was never trained; Dat+SEAM stand-in.
    ("YOLO11m-Pose+Data Augmentation+ECA+CBAM+C2PSA+SEAM*",
     W("yolo11m_srd_abl_c_e150_pat10_imgsz1280"), "srd"),
]

GROUP_B = [
    ("YOLO11m-Pose", W("yolo11m_srd_abl_a_e150_pat10_imgsz1280"), "srd"),
    ("YOLO11m-Pose+Data Augmentation", W("yolo11m_pose_baseline_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose+Data Augmentation+ECA", W("yolo11m_pose_b2_eca_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose+Data Augmentation+CBAM", W("yolo11m_pose_b4_cbam_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose+Data Augmentation+C2PSA",
     W("yolo11m_pose_extra_c2psa_sgd_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose+Data Augmentation+SEAM", W("yolo11m_srd_abl_c_e150_pat10_imgsz1280"), "srd"),
    ("YOLO11m-Pose+Data Augmentation+SRD", W("yolo11m_pose_srd_e150_pat10_imgsz1280"), "srd"),
    ("YOLO11m-Pose+Data Augmentation+PDCA", W("yolo11m_pose_pdca_e150_pat10_imgsz1280"), "attn"),
]


def gt_keypoint_visible(g) -> bool:
    """Paper eval excludes unlabeled keypoints (visibility==0 / coord 0,0)."""
    if not g.kpts:
        return False
    x, y, v = g.kpts[0][0], g.kpts[0][1], g.kpts[0][2]
    if v <= 0:
        return False
    if x == 0.0 and y == 0.0:
        return False
    return True


def eval_weights(label: str, weights: Path, kind: str) -> dict:
    register_attention_modules()
    if kind == "srd":
        register_srd_modules()

    # Prefer Attention Codes data yaml (same as training) if present
    data_yaml = Path("/home/raees/TUBITAK/Attention In YoloPosev11/Codes/CropsOrWeed9_yolopose.yaml")
    if not data_yaml.is_file():
        data_yaml = DATA_YAML

    img_dir, lbl_dir, names = load_dataset_info(data_yaml)
    print(f"\n=== {label} ===", flush=True)
    print(f"  data={data_yaml}  imgs={img_dir}", flush=True)

    model = YOLO(str(weights))
    n_gt = n_det = n_matched = 0
    n_box_matched_pred = 0
    dists: list[float] = []
    diag_sum = 0.0
    diag_n = 0
    t0 = time.time()

    for res in model.predict(
        source=str(img_dir),
        imgsz=IMGSZ,
        conf=EVAL_CONF,
        iou=EVAL_IOU,
        device="cpu",
        stream=True,
        verbose=False,
    ):
        im = cv2.imread(str(res.path))
        if im is None:
            continue
        h, w = im.shape[:2]
        diag_sum += float(np.hypot(w, h))
        diag_n += 1
        stem = Path(res.path).stem
        gt_all = parse_gt(lbl_dir / f"{stem}.txt", w, h)
        # Match paper: only visible / labeled stem keypoints contribute to CAR/MEA.
        gt_items = [g for g in gt_all if gt_keypoint_visible(g)]
        n_gt += len(gt_items)

        pred_items: list[PredItem] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            kxy = res.keypoints.xy.cpu().numpy() if res.keypoints is not None and res.keypoints.xy is not None else None
            kconf = res.keypoints.conf.cpu().numpy() if res.keypoints is not None and res.keypoints.conf is not None else None
            for i in range(len(xyxy)):
                kpts = []
                if kxy is not None and i < len(kxy):
                    for k in range(kxy.shape[1]):
                        x, y = float(kxy[i, k, 0]), float(kxy[i, k, 1])
                        v = float(kconf[i, k]) if kconf is not None else 1.0
                        if math.isfinite(x) and math.isfinite(y):
                            kpts.append([x, y, v])
                pred_items.append(
                    PredItem(cls=int(cls[i]), conf=float(confs[i]),
                             bbox=[float(v) for v in xyxy[i]], kpts=kpts)
                )
        n_det += len(pred_items)

        matches = greedy_loc_match(gt_items, pred_items, iou_thr=LOC_IOU_THR)
        n_box_matched_pred += len(matches)
        for gi, pi in matches:
            g, p = gt_items[gi], pred_items[pi]
            if g.kpts and p.kpts:
                d = float(np.hypot(g.kpts[0][0] - p.kpts[0][0], g.kpts[0][1] - p.kpts[0][1]))
                n_matched += 1
                dists.append(d)

        if diag_n % 100 == 0:
            print(f"  … {diag_n} images", flush=True)

    elapsed = time.time() - t0
    mean_diag = diag_sum / diag_n if diag_n else 2206.84
    arr = np.asarray(dists, dtype=np.float64)
    n_fp = n_det - n_matched          # predictions without a kpt-level assignment
    n_fn = n_gt - n_matched           # GT without a kpt-level assignment
    # Also report box-level unmatched preds
    n_fp_box = n_det - n_box_matched_pred

    car = 100.0 * n_matched / n_gt if n_gt else 0.0
    prec = 100.0 * n_matched / n_det if n_det else 0.0
    f1 = 200.0 * n_matched / (n_gt + n_det) if (n_gt + n_det) else 0.0
    fp_pct = 100.0 * n_fp / n_det if n_det else 0.0
    fn_pct = 100.0 * n_fn / n_gt if n_gt else 0.0
    mea = float(arr.mean()) if arr.size else float("nan")
    rmse_px = float(np.sqrt(np.mean(arr * arr))) if arr.size else float("nan")
    rmse_pct = 100.0 * rmse_px / mean_diag if math.isfinite(rmse_px) else float("nan")

    out = {
        "label": label,
        "weights": str(weights),
        "n_images": diag_n,
        "n_gt": n_gt,
        "n_det": n_det,
        "n_matched": n_matched,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "n_fp_box": n_fp_box,
        "FP_pct": round(fp_pct, 2),
        "FN_pct": round(fn_pct, 2),
        "CAR_pct": round(car, 2),
        "Prec_pct": round(prec, 2),
        "F1_pct": round(f1, 2),
        "MEA_px": round(mea, 3) if math.isfinite(mea) else None,
        "RMSE_px": round(rmse_px, 3) if math.isfinite(rmse_px) else None,
        "RMSE_pct": round(rmse_pct, 4) if math.isfinite(rmse_pct) else None,
        "mean_diag_px": round(mean_diag, 2),
        "elapsed_sec": round(elapsed, 1),
    }
    print(
        f"  FP%={out['FP_pct']:.2f} FN%={out['FN_pct']:.2f} "
        f"CAR={out['CAR_pct']:.2f} Prec={out['Prec_pct']:.2f} F1={out['F1_pct']:.2f} "
        f"MEA={out['MEA_px']} RMSE%={out['RMSE_pct']} ({elapsed:.0f}s)",
        flush=True,
    )
    del model
    return out


def main():
    unique: dict[str, tuple[str, str]] = {}
    for label, path, kind in GROUP_A + GROUP_B:
        unique[str(path)] = (label, kind)

    cache: dict[str, dict] = {}
    if PARTIAL.exists():
        try:
            prev = json.loads(PARTIAL.read_text())
            # Only resume entries that used the visible-GT protocol (MEA aligned to paper).
            for k, v in prev.get("partial_results", {}).items():
                if v.get("skip_invisible_gt") and v.get("CAR_pct") is not None:
                    cache[k] = v
            print(f"Resumed {len(cache)} models (visible-GT protocol)", flush=True)
        except Exception as e:
            print(f"Resume failed: {e}", flush=True)

    for path, (label, kind) in unique.items():
        if path in cache and cache[path].get("CAR_pct") is not None:
            print(f"SKIP (cached): {label}", flush=True)
            continue
        row = eval_weights(label, Path(path), kind)
        row["skip_invisible_gt"] = True
        cache[path] = row
        PARTIAL.write_text(json.dumps({"partial_results": cache}, indent=2))

    def table(group):
        rows = []
        for label, path, _ in group:
            m = dict(cache[str(path)])
            m["label"] = label
            rows.append(m)
        return rows

    out = {
        "conf": EVAL_CONF,
        "imgsz": IMGSZ,
        "nms_iou": EVAL_IOU,
        "loc_iou_thr": LOC_IOU_THR,
        "definitions": {
            "FP_pct": "100 * (n_det - n_matched) / n_det",
            "FN_pct": "100 * (n_gt - n_matched) / n_gt",
            "CAR_pct": "100 * n_matched / n_gt  (paper Correct Assignment Rate)",
            "Prec_pct": "100 * n_matched / n_det",
            "F1_pct": "200 * n_matched / (n_gt + n_det)",
            "MEA_px": "mean keypoint Euclidean error over matched pairs",
            "RMSE_pct": "100 * RMSE_px / mean_image_diagonal",
            "skip_invisible_gt": (
                "GT keypoints with visibility<=0 or (0,0) are excluded from n_gt and matching "
                "(82 unlabeled stems on this test set); required to match paper MEA/RMSE."
            ),
            "note_seam": (
                "Exact ECA+CBAM+C2PSA+SEAM was not trained; "
                "Dat+SEAM (srd_abl_c) used for that SEAM* column."
            ),
        },
        "group_cumulative": table(GROUP_A),
        "group_single_module": table(GROUP_B),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {OUT_JSON}", flush=True)

    def print_table(title, rows):
        print(f"\n{'=' * 110}")
        print(title)
        print(f"{'=' * 110}")
        hdr = (
            f"{'Method':58s} {'FP%':>7s} {'FN%':>7s} {'CAR%':>7s} "
            f"{'Prec':>7s} {'F1':>7s} {'MEA':>8s} {'RMSE%':>8s}"
        )
        print(hdr)
        print("-" * len(hdr))
        for m in rows:
            print(
                f"{m['label'][:58]:58s} "
                f"{m['FP_pct']:7.2f} {m['FN_pct']:7.2f} {m['CAR_pct']:7.2f} "
                f"{m['Prec_pct']:7.2f} {m['F1_pct']:7.2f} "
                f"{(m['MEA_px'] or float('nan')):8.3f} "
                f"{(m['RMSE_pct'] or float('nan')):8.4f}"
            )

    print_table("GROUP A — Cumulative (full test set)", out["group_cumulative"])
    print_table("GROUP B — Single-module (full test set)", out["group_single_module"])


if __name__ == "__main__":
    main()
