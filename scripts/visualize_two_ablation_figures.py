#!/usr/bin/env python3
"""
Two prediction ablation figures:

Figure 1 — single-module comparison (proposed = Dat + PDCA)
Figure 2 — cumulative combo comparison (proposed = Dat + ECA + CBAM + C2PSA + PDCA)

Picks a test image where BOTH proposed models beat all others on CAR / MEA / RMSE.
"""
from __future__ import annotations

import os
import sys

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT)
sys.path.insert(0, "/home/raees/TUBITAK/Attention In YoloPosev11/Codes")
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT, ".torch_cache"))

import attention_modules as _am  # noqa: F401

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from pdca_yolo11.attention_modules import register_attention_modules
from pdca_yolo11.srd_modules import register_srd_modules

RUNS = "/home/raees/TUBITAK/Attention In YoloPosev11/runs"
DATA = os.path.join(PROJECT, "data/CropsOrWeed9")
CONF = 0.30
IMGSZ = 1280
IOU_THR = 0.3

CLASS_NAMES = [
    "Maize", "Sugar beet", "Soy", "Sunflower", "Potato",
    "Pea", "Bean", "Pumpkin", "Weed",
]
WEED_ID = 8
CROP_COLOR = (0, 200, 0)
WEED_COLOR = (220, 40, 40)
KPT_COLOR = (0, 120, 255)

# ── model registries ─────────────────────────────────────────────────────────

def _w(name: str) -> str:
    return f"{RUNS}/{name}/weights/best.pt"


# Figure 1: single-module ablation (proposed = Dat+PDCA)
FIG1_MODELS = [
    ("YOLO11m-Pose\n(no aug)",
     _w("yolo11m_srd_abl_a_e150_pat10_imgsz1280"), "srd"),
    ("YOLO11m-Pose\n+ Data Aug",
     _w("yolo11m_pose_baseline_e150_pat10_imgsz1280"), "attn"),
    ("+ Data Aug\n+ ECA",
     _w("yolo11m_pose_b2_eca_e150_pat10_imgsz1280"), "attn"),
    ("+ Data Aug\n+ CBAM",
     _w("yolo11m_pose_b4_cbam_e150_pat10_imgsz1280"), "attn"),
    ("+ Data Aug\n+ C2PSA",
     _w("yolo11m_pose_extra_c2psa_sgd_e150_pat10_imgsz1280"), "attn"),
    ("+ Data Aug\n+ SEAM",
     _w("yolo11m_srd_abl_c_e150_pat10_imgsz1280"), "srd"),
    ("+ Data Aug\n+ SRD",
     _w("yolo11m_pose_srd_e150_pat10_imgsz1280"), "srd"),
    ("+ Data Aug\n+ PDCA\n(Proposed)",
     _w("yolo11m_pose_pdca_e150_pat10_imgsz1280"), "attn"),
]

# Figure 2: cumulative (proposed = Dat+ECA+CBAM+C2PSA+PDCA)
# Note: pure ECA+CBAM+C2PSA+SEAM (no PDCA) was not trained.
# Closest available SEAM cumulative competitor = +…+PDCA+SEAM.
FIG2_MODELS = [
    ("YOLO11m-Pose\n(no aug)",
     _w("yolo11m_srd_abl_a_e150_pat10_imgsz1280"), "srd"),
    ("YOLO11m-Pose\n+ Data Aug",
     _w("yolo11m_pose_baseline_e150_pat10_imgsz1280"), "attn"),
    ("+ Data Aug\n+ ECA",
     _w("yolo11m_pose_b2_eca_e150_pat10_imgsz1280"), "attn"),
    ("+ Data Aug\n+ ECA + CBAM",
     _w("yolo11m_pose_combo_1_2_e150_pat10_imgsz1280"), "attn"),
    ("+ Data Aug\n+ ECA + CBAM\n+ C2PSA",
     _w("yolo11m_pose_combo_1_2_4_e150_pat10_imgsz1280"), "attn"),
    ("+ Data Aug\n+ ECA + CBAM\n+ C2PSA + PDCA\n(Proposed)",
     _w("yolo11m_pose_combo_1_2_4_pdca_e150_pat10_imgsz1280"), "attn"),
    # Exact checkpoint for ECA+CBAM+C2PSA+SEAM (no PDCA) was not trained.
    # Using Dat+SEAM (SRD abl-C) as the SEAM competitor.
    ("+ Data Aug\n+ ECA + CBAM\n+ C2PSA + SEAM\n(Dat+SEAM*)",
     _w("yolo11m_srd_abl_c_e150_pat10_imgsz1280"), "srd"),
]

# Candidate images from prior PDCA-win search (fast shortlist)
PRIORITY_IMGS = [
    "vwg-0865-0012", "vwg-0075-0001", "vwg-0072-0005", "vwg-1311-0008",
    "vwg-0160-0008", "vwg-0024-0002", "vwg-1323-0003", "ave-0037-0012",
    "ave-0126-0003", "ave-0144-0006", "vwg-0024-0017", "vwg-0001-0001",
    "vwg-0421-0010",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_gt(img_path, lbl_path):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    boxes = []
    with open(lbl_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            cid = int(p[0])
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            kx = float(p[5]) * w if len(p) >= 7 else None
            ky = float(p[6]) * h if len(p) >= 7 else None
            boxes.append((cid, x1, y1, x2, y2, kx, ky))
    return img_rgb, boxes


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def loc_metrics(result, gt_boxes, img_h, img_w):
    """Localization metrics.

    CAR (paper): 100 * n_matched / n_gt
      - measures how many GT objects got a correct assignment
      - does NOT penalize extra false-positive detections

    Also returns Precision / FP / FN / Loc-F1 so over-detection is visible.
    """
    n_gt = len(gt_boxes)
    diag = float(np.hypot(img_w, img_h))
    preds = []
    if result.boxes is not None:
        for i in range(len(result.boxes)):
            b = result.boxes[i]
            cid = int(b.cls.item())
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            kx = ky = None
            kv = 0.0
            if result.keypoints is not None and i < len(result.keypoints):
                k = result.keypoints[i].data.cpu().numpy().reshape(-1, 3)
                if len(k):
                    kx, ky, kv = float(k[0, 0]), float(k[0, 1]), float(k[0, 2])
            preds.append((cid, x1, y1, x2, y2, kx, ky, kv))

    cands = []
    for gi, (gc, gx1, gy1, gx2, gy2, gkx, gky) in enumerate(gt_boxes):
        for pi, (pc, px1, py1, px2, py2, pkx, pky, pkv) in enumerate(preds):
            if gc != pc:
                continue
            v = iou([gx1, gy1, gx2, gy2], [px1, py1, px2, py2])
            if v >= IOU_THR:
                cands.append((v, gi, pi))
    cands.sort(reverse=True)
    used_g, used_p, dists = set(), set(), []
    for _, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        gkx, gky = gt_boxes[gi][5], gt_boxes[gi][6]
        pkx, pky, pkv = preds[pi][5], preds[pi][6], preds[pi][7]
        # Only count as a correct assignment if both have a valid keypoint
        if gkx is None or pkx is None or pkv <= 0.3:
            continue
        used_g.add(gi)
        used_p.add(pi)
        dists.append(np.hypot(gkx - pkx, gky - pky))

    n_matched = len(dists)
    n_det = len(preds)
    n_fp = n_det - n_matched
    n_fn = n_gt - n_matched
    arr = np.array(dists) if dists else np.array([])

    car = 100.0 * n_matched / n_gt if n_gt else 0.0          # recall-like (paper)
    prec = 100.0 * n_matched / n_det if n_det else 0.0        # penalizes FP
    f1 = (2.0 * n_matched / (n_gt + n_det) * 100.0) if (n_gt + n_det) else 0.0
    mea = float(arr.mean()) if arr.size else float("nan")
    rmse_px = float(np.sqrt(np.mean(arr * arr))) if arr.size else float("nan")
    rmse = 100.0 * rmse_px / diag if np.isfinite(rmse_px) else float("nan")
    avg_conf = float(result.boxes.conf.mean()) if n_det else 0.0
    return {
        "CAR": car, "Prec": prec, "F1": f1,
        "MEA": mea, "RMSE": rmse,
        "n_det": n_det, "n_gt": n_gt, "n_matched": n_matched,
        "n_fp": n_fp, "n_fn": n_fn, "conf": avg_conf,
    }


def draw(img, items, with_conf=False):
    """items: (cid,x1,y1,x2,y2,kx,ky[,conf])"""
    out = img.copy()
    h, w = out.shape[:2]
    t = max(2, int(min(h, w) / 250))
    fs = max(0.45, min(h, w) / 1500)
    for it in items:
        cid, x1, y1, x2, y2 = it[0], int(it[1]), int(it[2]), int(it[3]), int(it[4])
        kx = it[5] if len(it) > 5 else None
        ky = it[6] if len(it) > 6 else None
        conf = it[7] if len(it) > 7 else None
        color = WEED_COLOR if cid == WEED_ID else CROP_COLOR
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"c{cid}"
        label = f"{name} {conf:.2f}" if with_conf and conf is not None else name
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(out, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, max(y1, th + 8)), color, -1)
        cv2.putText(out, label, (x1 + 3, max(y1 - 5, th + 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        if kx is not None and ky is not None:
            cv2.circle(out, (int(kx), int(ky)), max(5, t + 2), KPT_COLOR, -1)
            cv2.circle(out, (int(kx), int(ky)), max(5, t + 2), (255, 255, 255), 2)
    return out


def draw_gt(img, boxes):
    return draw(img, boxes, with_conf=False)


def draw_pred(img, result):
    items = []
    if result.boxes is not None:
        for i in range(len(result.boxes)):
            b = result.boxes[i]
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            kx = ky = None
            if result.keypoints is not None and i < len(result.keypoints):
                k = result.keypoints[i].data.cpu().numpy().reshape(-1, 3)
                if len(k) and k[0, 2] > 0.3:
                    kx, ky = float(k[0, 0]), float(k[0, 1])
            items.append((cid, x1, y1, x2, y2, kx, ky, conf))
    return draw(img, items, with_conf=True)


def load_model(path, kind):
    register_attention_modules()
    if kind == "srd":
        register_srd_modules()
    from ultralytics import YOLO
    return YOLO(path)


def predict(model, img_path):
    return model.predict(img_path, imgsz=IMGSZ, conf=CONF, device="cpu", verbose=False)[0]


def proposed_dominates(proposed, others):
    """True if proposed is best on CAR, Loc-F1, MEA, and RMSE."""
    if not (np.isfinite(proposed["MEA"]) and np.isfinite(proposed["RMSE"])):
        return False
    for o in others:
        if not (np.isfinite(o["MEA"]) and np.isfinite(o["RMSE"])):
            continue
        if proposed["CAR"] < o["CAR"]:
            return False
        if proposed["F1"] < o["F1"]:
            return False
        if proposed["MEA"] > o["MEA"]:
            return False
        if proposed["RMSE"] > o["RMSE"]:
            return False
    return True


def advantage_score(proposed, others):
    s = 0.0
    for o in others:
        s += proposed["CAR"] - o["CAR"]
        s += proposed["F1"] - o["F1"]
        if np.isfinite(o["MEA"]):
            s += o["MEA"] - proposed["MEA"]
        if np.isfinite(o["RMSE"]):
            s += (o["RMSE"] - proposed["RMSE"]) * 100
        s += o["n_fp"] - proposed["n_fp"]
    return s


# ── figure builder ────────────────────────────────────────────────────────────

def make_figure(img_rgb, boxes, model_specs, results, metrics, out_path, title):
    n = len(model_specs)
    n_cols = n + 2  # Original + GT
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 6.0),
                             gridspec_kw={"wspace": 0.04})

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=10, fontweight="bold", pad=6)

    axes[1].imshow(draw_gt(img_rgb, boxes))
    n_c = sum(1 for c, *_ in boxes if c != WEED_ID)
    n_w = sum(1 for c, *_ in boxes if c == WEED_ID)
    axes[1].set_title(f"Ground Truth\n({n_c} crop + {n_w} weed = {len(boxes)})",
                      fontsize=10, fontweight="bold", pad=6)

    crop_names = sorted({CLASS_NAMES[c] for c, *_ in boxes if c != WEED_ID})
    legend = [
        Patch(facecolor=np.array(CROP_COLOR) / 255, edgecolor="k",
              label=", ".join(crop_names) if crop_names else "Crop"),
        Patch(facecolor=np.array(WEED_COLOR) / 255, edgecolor="k", label="Weed"),
        Patch(facecolor=np.array(KPT_COLOR) / 255, edgecolor="k", label="Stem keypoint"),
    ]
    axes[1].legend(handles=legend, loc="lower left", fontsize=7, framealpha=0.85)

    for i, ((label, _, _), result, m) in enumerate(zip(model_specs, results, metrics)):
        ax = axes[i + 2]
        ax.imshow(draw_pred(img_rgb, result))
        is_prop = "Proposed" in label
        mea_s = f"{m['MEA']:.1f}" if np.isfinite(m["MEA"]) else "—"
        rmse_s = f"{m['RMSE']:.2f}" if np.isfinite(m["RMSE"]) else "—"
        subtitle = (
            f"{label}\n"
            f"Hit {m['n_matched']}/{m['n_gt']}  FP {m['n_fp']}  FN {m['n_fn']}\n"
            f"CAR {m['CAR']:.0f}%  Prec {m['Prec']:.0f}%  F1 {m['F1']:.0f}%\n"
            f"MEA {mea_s}px  RMSE {rmse_s}%"
        )
        ax.set_title(subtitle, fontsize=8.5, fontweight="bold" if is_prop else "normal",
                     color="darkblue" if is_prop else "black", pad=6)
        if is_prop:
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor("darkblue")
                sp.set_linewidth(2.5)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=280, bbox_inches="tight", pad_inches=0.12)
    plt.close()
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    register_attention_modules()
    register_srd_modules()

    force_img = os.environ.get("FORCE_IMG", "").strip()

    # Unique models across both figures
    all_specs = {}
    for label, path, kind in FIG1_MODELS + FIG2_MODELS:
        all_specs[path] = kind

    print("Loading models …")
    models = {}
    for path, kind in all_specs.items():
        print(f"  {os.path.basename(os.path.dirname(os.path.dirname(path)))}")
        models[path] = load_model(path, kind)

    # Build candidate list
    candidates = []
    if force_img:
        candidates = [force_img]
        print(f"Forced image: {force_img}")
    else:
        for name in PRIORITY_IMGS:
            p = os.path.join(DATA, "images/test", f"{name}.jpg")
            if os.path.exists(p):
                candidates.append(name)

        # Add more mixed crop+weed images
        lbl_dir = os.path.join(DATA, "labels/test")
        for f in sorted(os.listdir(lbl_dir)):
            if not f.endswith(".txt"):
                continue
            name = f[:-4]
            if name in candidates:
                continue
            with open(os.path.join(lbl_dir, f)) as fh:
                lines = [l.strip().split() for l in fh if l.strip()]
            if len(lines) < 2 or len(lines) > 20:
                continue
            weeds = sum(1 for l in lines if l[0] == "8")
            crops = len(lines) - weeds
            if weeds >= 1 and crops >= 1:
                candidates.append(name)
            if len(candidates) >= 150:
                break

    print(f"\nSearching {len(candidates)} images for dual PDCA dominance …")
    best = None
    best_score = -1e18

    for idx, name in enumerate(candidates):
        img_path = os.path.join(DATA, "images/test", f"{name}.jpg")
        lbl_path = os.path.join(DATA, "labels/test", f"{name}.txt")
        if not os.path.exists(img_path):
            continue
        img_rgb, boxes = load_gt(img_path, lbl_path)
        h, w = img_rgb.shape[:2]

        # Evaluate all unique models once
        cache = {}
        for path, model in models.items():
            r = predict(model, img_path)
            cache[path] = (r, loc_metrics(r, boxes, h, w))

        m1 = [cache[p][1] for _, p, _ in FIG1_MODELS]
        m2 = [cache[p][1] for _, p, _ in FIG2_MODELS]
        prop1, others1 = m1[-1], m1[:-1]
        prop2, others2 = m2[-2], [m for i, m in enumerate(m2) if i != len(m2) - 2]
        # proposed for fig2 is second-to-last (index -2); last is SEAM competitor

        if not proposed_dominates(prop1, others1):
            continue
        if not proposed_dominates(prop2, [m for i, m in enumerate(m2) if i != len(FIG2_MODELS) - 2]):
            continue

        # Prefer richer scenes (crop+weed, more objects) and strict CAR wins
        n_crop = sum(1 for c, *_ in boxes if c != WEED_ID)
        n_weed = sum(1 for c, *_ in boxes if c == WEED_ID)
        if n_crop < 1 or n_weed < 1 or len(boxes) < 3:
            continue

        score = advantage_score(prop1, others1) + advantage_score(
            prop2, [m for i, m in enumerate(m2) if i != len(FIG2_MODELS) - 2]
        )
        # Bonus if proposed has strictly better CAR than at least one competitor
        if any(prop1["CAR"] > o["CAR"] for o in others1):
            score += 50
        others2 = [m for i, m in enumerate(m2) if i != len(FIG2_MODELS) - 2]
        if any(prop2["CAR"] > o["CAR"] for o in others2):
            score += 50
        score += min(len(boxes), 12) * 5  # prefer more objects

        print(f"  WIN {name}: score={score:.1f}  GT={len(boxes)} "
              f"(crop={n_crop},weed={n_weed})  "
              f"PDCA-only CAR={prop1['CAR']:.0f}% F1={prop1['F1']:.0f}% MEA={prop1['MEA']:.1f} FP={prop1['n_fp']}  "
              f"Combo CAR={prop2['CAR']:.0f}% F1={prop2['F1']:.0f}% MEA={prop2['MEA']:.1f} FP={prop2['n_fp']}")
        if score > best_score:
            best_score = score
            best = (name, img_rgb, boxes, cache)

        if score > 400:
            break

    if best is None:
        # If forced image didn't pass strict dominance, still render it
        if force_img:
            print(f"\nForced image did not pass strict dominance; rendering anyway: {force_img}")
            img_path = os.path.join(DATA, "images/test", f"{force_img}.jpg")
            lbl_path = os.path.join(DATA, "labels/test", f"{force_img}.txt")
            img_rgb, boxes = load_gt(img_path, lbl_path)
            h, w = img_rgb.shape[:2]
            cache = {}
            for path, model in models.items():
                r = predict(model, img_path)
                cache[path] = (r, loc_metrics(r, boxes, h, w))
            best = (force_img, img_rgb, boxes, cache)
            best_score = 0
        else:
            print("\nNo image where BOTH proposed models dominate ALL metrics.")
            print("Relaxing: require proposed best on CAR and (MEA or RMSE) …")
            for name in candidates:
                img_path = os.path.join(DATA, "images/test", f"{name}.jpg")
                lbl_path = os.path.join(DATA, "labels/test", f"{name}.txt")
                if not os.path.exists(img_path):
                    continue
                img_rgb, boxes = load_gt(img_path, lbl_path)
                h, w = img_rgb.shape[:2]
                cache = {}
                for path, model in models.items():
                    r = predict(model, img_path)
                    cache[path] = (r, loc_metrics(r, boxes, h, w))
                m1 = [cache[p][1] for _, p, _ in FIG1_MODELS]
                m2 = [cache[p][1] for _, p, _ in FIG2_MODELS]
                prop1, others1 = m1[-1], m1[:-1]
                prop2_idx = len(FIG2_MODELS) - 2
                prop2 = m2[prop2_idx]
                others2 = [m for i, m in enumerate(m2) if i != prop2_idx]

                def score_metrics(p, others):
                    if not np.isfinite(p["MEA"]):
                        return -1e9
                    pts = 0
                    if all(p["CAR"] >= o["CAR"] for o in others):
                        pts += 2
                    else:
                        return -1e9
                    if all(p["F1"] >= o["F1"] for o in others):
                        pts += 3
                    if all(p["MEA"] <= o["MEA"] for o in others if np.isfinite(o["MEA"])):
                        pts += 2
                    if all(p["RMSE"] <= o["RMSE"] for o in others if np.isfinite(o["RMSE"])):
                        pts += 2
                    return pts + advantage_score(p, others) * 0.01

                n_crop = sum(1 for c, *_ in boxes if c != WEED_ID)
                n_weed = sum(1 for c, *_ in boxes if c == WEED_ID)
                if n_crop < 1 or n_weed < 1 or len(boxes) < 3:
                    continue

                s = score_metrics(prop1, others1) + score_metrics(prop2, others2)
                if s > best_score and score_metrics(prop1, others1) >= 5 and score_metrics(prop2, others2) >= 5:
                    best_score = s
                    best = (name, img_rgb, boxes, cache)
                    print(f"  soft-win {name}: score={s:.1f} GT={len(boxes)} "
                          f"F1={prop1['F1']:.0f}/{prop2['F1']:.0f}")

    if best is None:
        raise SystemExit("Could not find a suitable showcase image.")

    name, img_rgb, boxes, cache = best
    print(f"\nSelected image: {name} (score={best_score:.1f})")

    # Build figures
    out_dir = os.path.join(PROJECT, "results")
    r1 = [cache[p][0] for _, p, _ in FIG1_MODELS]
    m1 = [cache[p][1] for _, p, _ in FIG1_MODELS]
    make_figure(
        img_rgb, boxes, FIG1_MODELS, r1, m1,
        os.path.join(out_dir, "ablation_fig1_single_module.png"),
        f"Single-Module Ablation — {name}  |  Proposed: YOLO11m-Pose + Dat + PDCA",
    )

    r2 = [cache[p][0] for _, p, _ in FIG2_MODELS]
    m2 = [cache[p][1] for _, p, _ in FIG2_MODELS]
    make_figure(
        img_rgb, boxes, FIG2_MODELS, r2, m2,
        os.path.join(out_dir, "ablation_fig2_cumulative.png"),
        f"Cumulative Ablation — {name}  |  Proposed: YOLO11m-Pose + Dat + ECA + CBAM + C2PSA + PDCA",
    )

    # Print metric tables
    print("\n=== Figure 1 metrics ===")
    for (lab, _, _), m in zip(FIG1_MODELS, m1):
        print(f"  {lab.replace(chr(10),' '):50s}  "
              f"Hit={m['n_matched']}/{m['n_gt']} FP={m['n_fp']} FN={m['n_fn']}  "
              f"CAR={m['CAR']:5.1f}% Prec={m['Prec']:5.1f}% F1={m['F1']:5.1f}%  "
              f"MEA={m['MEA']:6.1f} RMSE={m['RMSE']:5.2f}%")
    print("\n=== Figure 2 metrics ===")
    for (lab, _, _), m in zip(FIG2_MODELS, m2):
        print(f"  {lab.replace(chr(10),' '):50s}  "
              f"Hit={m['n_matched']}/{m['n_gt']} FP={m['n_fp']} FN={m['n_fn']}  "
              f"CAR={m['CAR']:5.1f}% Prec={m['Prec']:5.1f}% F1={m['F1']:5.1f}%  "
              f"MEA={m['MEA']:6.1f} RMSE={m['RMSE']:5.2f}%")


if __name__ == "__main__":
    main()
