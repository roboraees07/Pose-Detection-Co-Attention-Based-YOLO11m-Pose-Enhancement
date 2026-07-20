#!/usr/bin/env python3
"""Per-model 9-class detection confusion matrices for ablation studies.

Protocol matches Attention Codes/make_confusion_matrices.py:
  conf=0.30, imgsz=1280, match IoU≥0.50, row-normalized %, Missed column.
Each model is saved as its own PNG (+ JSON/CSV).
"""
from __future__ import annotations

import json
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
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from pdca_yolo11.attention_modules import register_attention_modules
from pdca_yolo11.srd_modules import register_srd_modules

RUNS = Path("/home/raees/TUBITAK/Attention In YoloPosev11/runs")
OUT_DIR = PROJECT / "results" / "confusion_matrices"
CONF = 0.30
IMGSZ = 1280
IOU_MATCH = 0.50
NMS_IOU = 0.7

CLASS_NAMES = [
    "Maize",
    "Sugar beet",
    "Soy",
    "Sunflower",
    "Potato",
    "Pea",
    "Bean",
    "Pumpkin",
    "Weed",
]
N = len(CLASS_NAMES)
MISSED_COL = "Missed"


def W(name: str) -> Path:
    return RUNS / name / "weights" / "best.pt"


# (title, slug, weights, kind)  kind in {"attn","srd"}
SINGLE = [
    ("YOLO11m-Pose (no augmentation)", "single_01_no_aug",
     W("yolo11m_srd_abl_a_e150_pat10_imgsz1280"), "srd"),
    ("YOLO11m-Pose + Data Augmentation", "single_02_dat",
     W("yolo11m_pose_baseline_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose + Dat + ECA", "single_03_eca",
     W("yolo11m_pose_b2_eca_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose + Dat + CBAM", "single_04_cbam",
     W("yolo11m_pose_b4_cbam_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose + Dat + C2PSA", "single_05_c2psa",
     W("yolo11m_pose_extra_c2psa_sgd_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose + Dat + SEAM", "single_06_seam",
     W("yolo11m_srd_abl_c_e150_pat10_imgsz1280"), "srd"),
    ("YOLO11m-Pose + Dat + SRD", "single_07_srd",
     W("yolo11m_pose_srd_e150_pat10_imgsz1280"), "srd"),
    ("YOLO11m-Pose + Dat + PDCA", "single_08_pdca",
     W("yolo11m_pose_pdca_e150_pat10_imgsz1280"), "attn"),
]

CUMULATIVE = [
    ("YOLO11m-Pose (no augmentation)", "cumul_01_no_aug",
     W("yolo11m_srd_abl_a_e150_pat10_imgsz1280"), "srd"),
    ("YOLO11m-Pose + Data Augmentation", "cumul_02_dat",
     W("yolo11m_pose_baseline_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose + Dat + ECA", "cumul_03_eca",
     W("yolo11m_pose_b2_eca_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose + Dat + ECA + CBAM", "cumul_04_eca_cbam",
     W("yolo11m_pose_combo_1_2_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose + Dat + ECA + CBAM + C2PSA", "cumul_05_eca_cbam_c2psa",
     W("yolo11m_pose_combo_1_2_4_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose + Dat + ECA + CBAM + C2PSA + PDCA (Proposed)",
     "cumul_06_eca_cbam_c2psa_pdca",
     W("yolo11m_pose_combo_1_2_4_pdca_e150_pat10_imgsz1280"), "attn"),
    ("YOLO11m-Pose + Dat + SEAM (competitor)", "cumul_07_seam",
     W("yolo11m_srd_abl_c_e150_pat10_imgsz1280"), "srd"),
]


def resolve_dirs() -> tuple[Path, Path]:
    data_yaml = Path("/home/raees/TUBITAK/Attention In YoloPosev11/Codes/CropsOrWeed9_yolopose.yaml")
    import yaml

    y = yaml.safe_load(data_yaml.read_text())
    root = Path(y["path"]).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    else:
        root = root.resolve()
    img_dir = (root / y["test"]).resolve() if not Path(y["test"]).is_absolute() else Path(y["test"]).resolve()
    parts = list(img_dir.parts)
    idx = parts.index("images")
    lbl_parts = parts.copy()
    lbl_parts[idx] = "labels"
    lbl_dir = Path(*lbl_parts)
    return img_dir, lbl_dir


def parse_gt(label_path: Path, w: int, h: int) -> list[dict]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        p = line.strip().split()
        if len(p) < 8:
            continue
        # skip unlabeled keypoints (visibility <= 0)
        if int(float(p[7])) <= 0:
            continue
        cls = int(float(p[0]))
        if cls < 0 or cls >= N:
            continue
        cx, cy = float(p[1]) * w, float(p[2]) * h
        bw, bh = float(p[3]) * w, float(p[4]) * h
        out.append({
            "cls": cls,
            "bbox": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
        })
    return out


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - inter
    )
    return float(inter / ua) if ua > 0 else 0.0


def extract_preds(result) -> list[dict]:
    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    boxes = result.boxes.xyxy.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    for i, c in enumerate(clss):
        if int(c) < 0 or int(c) >= N:
            continue
        out.append({"cls": int(c), "bbox": boxes[i].tolist()})
    return out


def accumulate_image(cm: np.ndarray, gts: list[dict], preds: list[dict]) -> tuple[int, int]:
    used: set[int] = set()
    for g in gts:
        ti = g["cls"]
        best_i, best_iou = -1, 0.0
        for i, p in enumerate(preds):
            if i in used:
                continue
            iou = iou_xyxy(g["bbox"], p["bbox"])
            if iou >= IOU_MATCH and iou > best_iou:
                best_iou, best_i = iou, i
        if best_i < 0:
            cm[ti, N] += 1
        else:
            used.add(best_i)
            cm[ti, preds[best_i]["cls"]] += 1
    n_fp = sum(1 for i in range(len(preds)) if i not in used)
    return len(gts), n_fp


def row_percent(cm: np.ndarray) -> np.ndarray:
    pct = np.zeros_like(cm, dtype=np.float64)
    for i in range(N):
        row_sum = cm[i].sum()
        if row_sum > 0:
            pct[i] = 100.0 * cm[i] / row_sum
    return pct


def plot_matrix(pct: np.ndarray, counts: np.ndarray, title: str, out_path: Path):
    col_labels = CLASS_NAMES + [MISSED_COL]
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(N + 1))
    ax.set_yticks(range(N))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_title(
        f"{title}\nRow-normalized detection confusion (%), "
        f"IoU≥{IOU_MATCH}, conf={CONF}",
        fontsize=12,
    )

    for i in range(N):
        for j in range(N + 1):
            val = pct[i, j]
            cnt = int(counts[i, j])
            txt = f"{val:.1f}%\n(n={cnt})"
            color = "white" if val > 55 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)

    for i in range(N):
        total = int(counts[i].sum())
        ax.text(N + 0.35, i, f"N={total}", va="center", fontsize=8, color="dimgray")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Row %", rotation=90)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_csv(pct: np.ndarray, counts: np.ndarray, path: Path):
    col_labels = CLASS_NAMES + [MISSED_COL]
    lines = ["True\\Pred," + ",".join(col_labels)]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(name + "," + ",".join(f"{pct[i, j]:.2f}" for j in range(N + 1)))
    path.with_suffix(".csv").write_text("\n".join(lines) + "\n")
    lines2 = ["True\\Pred," + ",".join(col_labels)]
    for i, name in enumerate(CLASS_NAMES):
        lines2.append(name + "," + ",".join(str(int(counts[i, j])) for j in range(N + 1)))
    path.with_name(path.stem + "_counts.csv").write_text("\n".join(lines2) + "\n")


def eval_model_cm(weights: Path, kind: str, img_dir: Path, lbl_dir: Path, device: str) -> dict:
    register_attention_modules()
    if kind == "srd":
        register_srd_modules()

    model = YOLO(str(weights))
    cm = np.zeros((N, N + 1), dtype=np.int64)
    total_gt = 0
    total_fp = 0
    n_img = 0
    t0 = time.time()

    for res in model.predict(
        source=str(img_dir),
        imgsz=IMGSZ,
        conf=CONF,
        iou=NMS_IOU,
        device=device,
        stream=True,
        verbose=False,
    ):
        im = cv2.imread(str(res.path))
        if im is None:
            continue
        h, w = im.shape[:2]
        gts = parse_gt(lbl_dir / f"{Path(res.path).stem}.txt", w, h)
        preds = extract_preds(res)
        ng, nfp = accumulate_image(cm, gts, preds)
        total_gt += ng
        total_fp += nfp
        n_img += 1
        if n_img % 200 == 0:
            print(f"    … {n_img} images", flush=True)

    pct = row_percent(cm)
    diag = int(np.trace(cm[:, :N]))
    matched = int(cm[:, :N].sum())
    acc_matched = 100.0 * diag / matched if matched else 0.0
    overall_correct = 100.0 * diag / total_gt if total_gt else 0.0

    return {
        "counts": cm.tolist(),
        "row_percent": pct.tolist(),
        "n_images": n_img,
        "total_gt": total_gt,
        "total_matched": matched,
        "total_correct_class": diag,
        "total_missed": int(cm[:, N].sum()),
        "total_false_positives": total_fp,
        "accuracy_on_matched_pct": round(acc_matched, 2),
        "overall_top1_correct_pct": round(overall_correct, 2),
        "elapsed_sec": round(time.time() - t0, 1),
        "row_labels": CLASS_NAMES,
        "col_labels": CLASS_NAMES + [MISSED_COL],
        "conf": CONF,
        "iou_match": IOU_MATCH,
        "imgsz": IMGSZ,
    }


def main():
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "0"
    except Exception:
        pass

    img_dir, lbl_dir = resolve_dirs()
    print(f"device={device}", flush=True)
    print(f"images={img_dir}", flush=True)
    print(f"labels={lbl_dir}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    single_dir = OUT_DIR / "single_module"
    cumul_dir = OUT_DIR / "cumulative"
    single_dir.mkdir(exist_ok=True)
    cumul_dir.mkdir(exist_ok=True)

    # Unique by weights path — compute once, write for every listing that shares it.
    jobs: dict[str, dict] = {}
    for group, out_root in ((SINGLE, single_dir), (CUMULATIVE, cumul_dir)):
        for title, slug, wp, kind in group:
            if not wp.is_file():
                raise FileNotFoundError(wp)
            key = str(wp)
            if key not in jobs:
                jobs[key] = {"title": title, "kind": kind, "wp": wp, "outputs": []}
            jobs[key]["outputs"].append(
                {"slug": slug, "png": out_root / f"confmat_{slug}_percent.png", "title": title}
            )

    summary = {
        "single_module": {},
        "cumulative": {},
        "protocol": {
            "conf": CONF,
            "imgsz": IMGSZ,
            "iou_match": IOU_MATCH,
            "nms_iou": NMS_IOU,
            "scope": "overall 9 classes (detection matching)",
            "normalization": "row-normalized % of true class",
        },
    }

    cache_json = OUT_DIR / "_cache_by_weights.json"
    cache: dict = {}
    if cache_json.exists():
        try:
            cache = json.loads(cache_json.read_text())
        except Exception:
            cache = {}

    for key, job in jobs.items():
        title, kind, wp = job["title"], job["kind"], job["wp"]
        print(f"\n=== {title} ===", flush=True)
        print(f"  weights={wp.name}", flush=True)
        if key in cache and cache[key].get("counts"):
            stats = cache[key]
            print("  SKIP compute (cached)", flush=True)
        else:
            stats = eval_model_cm(wp, kind, img_dir, lbl_dir, device)
            stats["model"] = title
            stats["weights"] = key
            cache[key] = stats
            cache_json.write_text(json.dumps(cache, indent=2))
            print(
                f"  GT={stats['total_gt']} correct={stats['total_correct_class']} "
                f"missed={stats['total_missed']} FP={stats['total_false_positives']} "
                f"overall={stats['overall_top1_correct_pct']}% ({stats['elapsed_sec']}s)",
                flush=True,
            )

        counts = np.array(stats["counts"], dtype=np.int64)
        pct = np.array(stats["row_percent"], dtype=np.float64)

        for out in job["outputs"]:
            slug = out["slug"]
            png_path = out["png"]
            plot_title = out["title"]
            plot_matrix(pct, counts, plot_title, png_path)
            stem = png_path.with_name(png_path.name.replace("_percent.png", ""))
            json_path = stem.with_suffix(".json")
            payload = {**stats, "model": plot_title, "slug": slug}
            json_path.write_text(json.dumps(payload, indent=2))
            save_csv(pct, counts, stem)

            bucket = "single_module" if slug.startswith("single_") else "cumulative"
            summary[bucket][slug] = {
                "model": plot_title,
                "png": str(png_path),
                "overall_top1_correct_pct": stats["overall_top1_correct_pct"],
                "accuracy_on_matched_pct": stats["accuracy_on_matched_pct"],
                "total_gt": stats["total_gt"],
                "total_missed": stats["total_missed"],
                "total_false_positives": stats["total_false_positives"],
            }
            print(f"  saved {png_path.name}", flush=True)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone → {OUT_DIR}", flush=True)
    print("Single-module PNGs:", len(list(single_dir.glob("*_percent.png"))))
    print("Cumulative PNGs:", len(list(cumul_dir.glob("*_percent.png"))))


if __name__ == "__main__":
    main()
