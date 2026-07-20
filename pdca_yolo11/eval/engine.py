"""Evaluation engine — weed class and 9-class overall metrics."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.metrics import ap_per_class, kpt_iou

from pdca_yolo11.constants import EVAL_IOU, IMGSZ, LOC_IOU_THR, WEED_CLASS_ID
from pdca_yolo11.paths import DATA_YAML


@dataclass
class GTItem:
    cls: int
    bbox: list[float]
    kpts: list[list[float]]


@dataclass
class PredItem:
    cls: int
    conf: float
    bbox: list[float]
    kpts: list[list[float]]


def load_dataset_info(yaml_path: Path) -> tuple[Path, Path, dict[int, str]]:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", "")).expanduser()
    if not root.is_absolute():
        root = (yaml_path.resolve().parent / root).resolve()
    test_rel = cfg.get("test", cfg.get("val"))
    if test_rel is None:
        raise ValueError(f"'test' or 'val' missing in {yaml_path}")
    img_dir = (root / test_rel).resolve() if not Path(test_rel).is_absolute() else Path(test_rel).resolve()
    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    parts = list(img_dir.parts)
    try:
        idx = parts.index("images")
        lbl_parts = parts.copy()
        lbl_parts[idx] = "labels"
        lbl_dir = Path(*lbl_parts)
    except ValueError:
        lbl_dir = img_dir.parent / "labels" / img_dir.name
    names_cfg = cfg.get("names", {})
    if isinstance(names_cfg, dict):
        names = {int(k): str(v) for k, v in names_cfg.items()}
    else:
        names = {i: str(v) for i, v in enumerate(names_cfg)}
    return img_dir, lbl_dir, names


def clip_box(x1, y1, x2, y2, w, h) -> list[float]:
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def parse_gt(label_path: Path, w: int, h: int) -> list[GTItem]:
    if not label_path.exists():
        return []
    items: list[GTItem] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) < 5:
            continue
        cls = int(vals[0])
        cx, cy, bw, bh = vals[1:5]
        bbox = clip_box((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, w, h)
        kpts: list[list[float]] = []
        for i in range(5, len(vals) - 2, 3):
            kx, ky, kv = vals[i] * w, vals[i + 1] * h, vals[i + 2]
            if math.isfinite(kx) and math.isfinite(ky):
                kpts.append([float(kx), float(ky), float(kv)])
        items.append(GTItem(cls=cls, bbox=bbox, kpts=kpts))
    return items


def iou_xyxy_np(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = aa + bb - inter
    return float(inter / denom) if denom > 0 else 0.0


def greedy_loc_match(gt_items, pr_items, iou_thr=0.1) -> list[tuple[int, int]]:
    cand = []
    for gi, g in enumerate(gt_items):
        for pi, p in enumerate(pr_items):
            if g.cls != p.cls:
                continue
            iou = iou_xyxy_np(g.bbox, p.bbox)
            if iou >= iou_thr:
                cand.append((iou, gi, pi))
    cand.sort(reverse=True)
    used_g, used_p, matches = set(), set(), []
    for _, gi, pi in cand:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi))
    return matches


def box_iou_torch(box1, box2):
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]))
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0))[:, None]
    area2 = ((box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0))[None, :]
    return inter / (area1 + area2 - inter).clamp(min=1e-9)


def match_predictions(pred_classes, true_classes, iou, iouv):
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    if pred_classes.numel() == 0 or true_classes.numel() == 0:
        return correct
    correct_class = true_classes[:, None] == pred_classes
    iou_np = (iou * correct_class).cpu().numpy()
    for i, thr in enumerate(iouv.cpu().tolist()):
        matches = np.array(np.nonzero(iou_np >= thr)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def to_full_by_class(ap_class, vec, ncls):
    out = np.zeros(ncls, dtype=np.float64)
    for i, c in enumerate(ap_class.astype(int)):
        if 0 <= c < ncls:
            out[c] = float(vec[i])
    return out


def ap_bundle(tp, conf, pred_cls, tgt_cls, names, ncls):
    if tgt_cls.size:
        _, _, p, r, f1, ap, ap_class, *_ = ap_per_class(tp, conf, pred_cls, tgt_cls, plot=False, names=names, prefix="")
    else:
        p = r = f1 = np.zeros((0,), dtype=np.float64)
        ap = np.zeros((0, 10), dtype=np.float64)
        ap_class = np.zeros((0,), dtype=np.int64)
    return {
        "p": to_full_by_class(ap_class, p, ncls),
        "r": to_full_by_class(ap_class, r, ncls),
        "f1": to_full_by_class(ap_class, f1, ncls),
        "ap50": to_full_by_class(ap_class, ap[:, 0] if ap.size else np.array([]), ncls),
        "ap5095": to_full_by_class(ap_class, ap.mean(1) if ap.size else np.array([]), ncls),
    }


def loc_stats(rec: dict) -> dict[str, float]:
    arr = np.array(rec["dists"], dtype=np.float32)
    n_gt, n_m = int(rec["n_gt"]), int(rec["n_matched"])
    return {
        "CAR_pct": (100.0 * n_m / n_gt) if n_gt else float("nan"),
        "MEA_px": float(arr.mean()) if arr.size else float("nan"),
        "RMSE_px": float(np.sqrt(np.mean(arr * arr))) if arr.size else float("nan"),
    }


def class_metrics(det, pose, loc, cid: int) -> dict[str, float]:
    return {
        "det_precision": float(det["p"][cid]),
        "det_recall": float(det["r"][cid]),
        "det_f1": float(det["f1"][cid]),
        "det_mAP50": float(det["ap50"][cid]),
        "det_mAP50_95": float(det["ap5095"][cid]),
        "pose_precision": float(pose["p"][cid]),
        "pose_recall": float(pose["r"][cid]),
        "pose_f1": float(pose["f1"][cid]),
        "pose_mAP50": float(pose["ap50"][cid]),
        "pose_mAP50_95": float(pose["ap5095"][cid]),
        "MEA_px": float(loc["MEA_px"]),
        "CAR_pct": float(loc["CAR_pct"]),
        "RMSE_px": float(loc["RMSE_px"]),
    }


def pct(x: float) -> float:
    return round(x * 100.0, 2) if math.isfinite(x) else float("nan")


def table_row_from_metrics(m: dict[str, float], mean_time_ms: float, rmse_pct: float) -> dict:
    return {
        "Det P(%)": pct(m["det_precision"]),
        "Det R(%)": pct(m["det_recall"]),
        "Det F1(%)": pct(m["det_f1"]),
        "Det mAP50(%)": pct(m["det_mAP50"]),
        "Det mAP50-95(%)": pct(m["det_mAP50_95"]),
        "Pose P(%)": pct(m["pose_precision"]),
        "Pose R(%)": pct(m["pose_recall"]),
        "Pose F1(%)": pct(m["pose_f1"]),
        "Pose mAP50(%)": pct(m["pose_mAP50"]),
        "Pose mAP50-95(%)": pct(m["pose_mAP50_95"]),
        "MEA(px)": round(m["MEA_px"], 3),
        "CAR(%)": round(m["CAR_pct"], 3),
        "RMSE(%)": round(rmse_pct, 4),
        "mean time(ms/img)": round(mean_time_ms, 2),
    }


def model_info(weights: Path, run_dir: Path | None) -> dict[str, Any]:
    n_params, layers, gflops = 0, 0, 0.0
    try:
        m = YOLO(str(weights))
        n_params = sum(p.numel() for p in m.model.parameters())
        layers = len(m.model.model)
    except Exception:
        pass
    if run_dir and (run_dir / "results.csv").is_file():
        log = "\n".join((run_dir / "results.csv").read_text(errors="ignore").splitlines()[:3])
    else:
        log = ""
    for path in [run_dir / "train.log" if run_dir else None, run_dir / ".." / f"train_{run_dir.name}.log" if run_dir else None]:
        if path and path.is_file():
            log = path.read_text(errors="ignore")
            break
    mm = re.search(r"summary:\s*(\d+)\s*layers,\s*([\d,]+)\s*parameters.*?([\d.]+)\s*GFLOPs", log)
    if mm:
        layers, n_params, gflops = int(mm.group(1)), int(mm.group(2).replace(",", "")), float(mm.group(3))
    return {"Parameters": n_params, "Params/M": round(n_params / 1e6, 2), "Layers": layers, "GFLOPs": gflops}


def epoch_stops(run_dir: Path) -> str:
    rc = run_dir / "results.csv"
    if not rc.is_file():
        return "?"
    lines = [ln for ln in rc.read_text().strip().splitlines() if ln and not ln.startswith("epoch")]
    if not lines:
        return "?"
    last_ep = int(lines[-1].split(",")[0])
    return str(last_ep)


def evaluate(
    weights: str | Path,
    *,
    data_yaml: Path = DATA_YAML,
    imgsz: int = IMGSZ,
    conf_thr: float = 0.30,
    iou_thr: float = EVAL_IOU,
    device: str = "0",
    loc_iou_thr: float = LOC_IOU_THR,
    weed_class_id: int = WEED_CLASS_ID,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full test-set evaluation; returns weed + overall metrics."""
    img_dir, lbl_dir, names = load_dataset_info(data_yaml)
    ncls = len(names)
    model = YOLO(str(weights))
    iouv = torch.linspace(0.5, 0.95, 10)

    stats_det, stats_pose = [], []
    per_cls_loc = {cid: {"n_gt": 0, "n_matched": 0, "dists": []} for cid in sorted(names)}
    diag_sum = 0.0
    diag_n = 0
    inf_ms_all: list[float] = []

    for res in model.predict(source=str(img_dir), imgsz=imgsz, conf=conf_thr, iou=iou_thr, device=device, stream=True, verbose=False):
        im = cv2.imread(str(res.path))
        if im is None:
            continue
        h, w = im.shape[:2]
        diag_sum += float(np.hypot(w, h))
        diag_n += 1
        stem = Path(res.path).stem
        gt_items = parse_gt(lbl_dir / f"{stem}.txt", w, h)
        for g in gt_items:
            if g.cls in per_cls_loc:
                per_cls_loc[g.cls]["n_gt"] += 1

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
                pred_items.append(PredItem(cls=int(cls[i]), conf=float(confs[i]), bbox=[float(v) for v in xyxy[i]], kpts=kpts))

        if gt_items:
            gt_cls = torch.tensor([g.cls for g in gt_items], dtype=torch.int64)
            gt_boxes = torch.tensor([g.bbox for g in gt_items], dtype=torch.float32)
            tgt_cls_np = gt_cls.numpy()
        else:
            gt_cls = torch.zeros((0,), dtype=torch.int64)
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            tgt_cls_np = np.array([], dtype=np.int64)

        if pred_items:
            pr_cls = torch.tensor([p.cls for p in pred_items], dtype=torch.int64)
            pr_boxes = torch.tensor([p.bbox for p in pred_items], dtype=torch.float32)
            pr_conf = np.array([p.conf for p in pred_items], dtype=np.float32)
        else:
            pr_cls = torch.zeros((0,), dtype=torch.int64)
            pr_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pr_conf = np.array([], dtype=np.float32)

        if gt_cls.numel() == 0 or pr_cls.numel() == 0:
            tp_det = np.zeros((pr_cls.shape[0], iouv.shape[0]), dtype=bool)
        else:
            tp_det = match_predictions(pr_cls, gt_cls, box_iou_torch(gt_boxes, pr_boxes), iouv)
        stats_det.append((tp_det, pr_conf, pr_cls.numpy(), tgt_cls_np))

        valid_gt = [i for i, g in enumerate(gt_items) if g.kpts]
        valid_pr = [i for i, p in enumerate(pred_items) if p.kpts]
        if not valid_gt or not valid_pr:
            tp_pose = np.zeros((pr_cls.shape[0], iouv.shape[0]), dtype=bool)
        else:
            gt_cls_pose = torch.tensor([gt_items[i].cls for i in valid_gt], dtype=torch.int64)
            pr_cls_pose = torch.tensor([pred_items[i].cls for i in valid_pr], dtype=torch.int64)
            gt_boxes_pose = torch.tensor([gt_items[i].bbox for i in valid_gt], dtype=torch.float32)
            gt_kpts_pose = torch.tensor([[[gt_items[i].kpts[0][0], gt_items[i].kpts[0][1], 1.0]] for i in valid_gt], dtype=torch.float32)
            pr_kpts_pose = torch.tensor([[[pred_items[i].kpts[0][0], pred_items[i].kpts[0][1], 1.0]] for i in valid_pr], dtype=torch.float32)
            area = ops.xyxy2xywh(gt_boxes_pose)[:, 2:].prod(1) * 0.53
            iou_pose = kpt_iou(gt_kpts_pose, pr_kpts_pose, sigma=np.ones(1, dtype=np.float32), area=area)
            tp_pose_small = match_predictions(pr_cls_pose, gt_cls_pose, iou_pose, iouv)
            tp_pose = np.zeros((pr_cls.shape[0], iouv.shape[0]), dtype=bool)
            for j, opi in enumerate(valid_pr):
                tp_pose[opi] = tp_pose_small[j]
        stats_pose.append((tp_pose, pr_conf, pr_cls.numpy(), tgt_cls_np))

        for gi, pi in greedy_loc_match(gt_items, pred_items, iou_thr=loc_iou_thr):
            g, p = gt_items[gi], pred_items[pi]
            if g.kpts and p.kpts:
                d = float(np.hypot(g.kpts[0][0] - p.kpts[0][0], g.kpts[0][1] - p.kpts[0][1]))
                per_cls_loc[g.cls]["n_matched"] += 1
                per_cls_loc[g.cls]["dists"].append(d)

        if res.speed.get("inference") is not None:
            inf_ms_all.append(float(res.speed["inference"]))

    det_tp = np.concatenate([s[0] for s in stats_det], 0) if stats_det else np.zeros((0, 10), dtype=bool)
    det_conf = np.concatenate([s[1] for s in stats_det], 0) if stats_det else np.array([])
    det_pr_cls = np.concatenate([s[2] for s in stats_det], 0) if stats_det else np.array([])
    det_tgt_cls = np.concatenate([s[3] for s in stats_det], 0) if stats_det else np.array([])

    pose_tp = np.concatenate([s[0] for s in stats_pose], 0) if stats_pose else np.zeros((0, 10), dtype=bool)
    pose_conf = np.concatenate([s[1] for s in stats_pose], 0) if stats_pose else np.array([])
    pose_pr_cls = np.concatenate([s[2] for s in stats_pose], 0) if stats_pose else np.array([])
    pose_tgt_cls = np.concatenate([s[3] for s in stats_pose], 0) if stats_pose else np.array([])

    det = ap_bundle(det_tp, det_conf, det_pr_cls, det_tgt_cls, names, ncls)
    pose = ap_bundle(pose_tp, pose_conf, pose_pr_cls, pose_tgt_cls, names, ncls)

    per_class = []
    for cid in sorted(names):
        loc = loc_stats(per_cls_loc[cid])
        per_class.append({"class_id": cid, "class_name": names[cid], **class_metrics(det, pose, loc, cid)})

    overall_loc = loc_stats({"n_gt": sum(per_cls_loc[c]["n_gt"] for c in per_cls_loc), "n_matched": sum(per_cls_loc[c]["n_matched"] for c in per_cls_loc), "dists": [d for c in per_cls_loc for d in per_cls_loc[c]["dists"]]})
    overall_m = {
        "det_precision": float(det["p"].mean()),
        "det_recall": float(det["r"].mean()),
        "det_f1": float(det["f1"].mean()),
        "det_mAP50": float(det["ap50"].mean()),
        "det_mAP50_95": float(det["ap5095"].mean()),
        "pose_precision": float(pose["p"].mean()),
        "pose_recall": float(pose["r"].mean()),
        "pose_f1": float(pose["f1"].mean()),
        "pose_mAP50": float(pose["ap50"].mean()),
        "pose_mAP50_95": float(pose["ap5095"].mean()),
        **{k: overall_loc[k] for k in ("MEA_px", "CAR_pct", "RMSE_px")},
    }
    weed_m = next(r for r in per_class if r["class_id"] == weed_class_id)
    mean_diag = diag_sum / diag_n if diag_n else 2206.84
    mean_time = float(np.mean(inf_ms_all)) if inf_ms_all else float("nan")

    def rmse_pct(m):
        return 100.0 * m["RMSE_px"] / mean_diag if math.isfinite(m["RMSE_px"]) else float("nan")

    info = model_info(Path(weights), run_dir)
    fps = round(1000.0 / mean_time, 1) if mean_time and mean_time > 0 else float("nan")

    return {
        "weights": str(weights),
        "predict_conf": conf_thr,
        "weed_class_id": weed_class_id,
        "mean_image_diagonal_px": mean_diag,
        "weed_metrics": {**weed_m, "mean_time_ms_per_image": mean_time, "RMSE_pct": rmse_pct(weed_m)},
        "overall_metrics": {**overall_m, "mean_time_ms_per_image": mean_time, "RMSE_pct": rmse_pct(overall_m)},
        "weed_table_row": table_row_from_metrics(weed_m, mean_time, rmse_pct(weed_m)),
        "overall_table_row": table_row_from_metrics(overall_m, mean_time, rmse_pct(overall_m)),
        "model_info": {**info, "FPS": fps, "mean_time_ms_per_image": mean_time},
        "epoch_stops": epoch_stops(run_dir) if run_dir else "?",
    }


def save_eval_json(result: dict, out_path: Path) -> None:
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
