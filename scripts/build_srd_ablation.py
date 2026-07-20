#!/usr/bin/env python3
"""Build YOLO11m-Pose SRD ablation YAMLs (SEAM × RVB × DWR), paper-style A–I (Dat = train flag)."""

from __future__ import annotations

import argparse
from pathlib import Path

CODES = Path(__file__).resolve().parent
OUT = CODES.parent / "configs" / "srd" / "ablation"

HEADER = """# SRD ablation {tag}: SEAM={seam} RVB={rvb} DWR={dwr} (YOLO11m-Pose, CropsOrWeed9)
nc: 9
kpt_shape: [1, 3]
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
"""

# Paper-mapped repeat counts on YOLO11m (Liu et al. / SRD-YOLO → YOLO11).
RVB_N = (3, 6, 3, 3, 3, 3)  # bb×2 + neck×4
DWR_N = (6, 3)  # bb P4, bb P5
STOCK_N = 2


def _c3(rvb: bool, dwr: bool, slot: str, ch: int, shortcut=False, e=0.25) -> str:
    """Backbone/neck block: RVB slot, DWR slot, or stock C3k2."""
    sc = "True" if shortcut else "False"
    if slot == "rvb":
        n = RVB_N[0] if ch <= 256 else RVB_N[1]
        if rvb:
            return f"  - [-1, {n}, C3k2_RVB, [{ch}, {sc}, {e}]]"
        return f"  - [-1, {STOCK_N}, C3k2, [{ch}, {sc}, {e}]]"
    if slot == "dwr":
        n = DWR_N[0] if ch <= 512 else DWR_N[1]
        if dwr:
            return f"  - [-1, {n}, C3k2_DWR, [{ch}, {sc}]]"
        if rvb:
            return f"  - [-1, {STOCK_N}, C3k2_RVB, [{ch}, {sc}, {e}]]"
        return f"  - [-1, {STOCK_N}, C3k2, [{ch}, {sc}]]"
    # neck
    idx = {"p4": 2, "p3": 3, "p4d": 4, "p5": 5}[slot]
    n = RVB_N[idx]
    if rvb:
        return f"  - [-1, {n}, C3k2_RVB, [{ch}, {str(slot == 'p5')}]]"
    return f"  - [-1, {STOCK_N}, C3k2, [{ch}, {str(slot == 'p5')}]]"


def build_yaml(seam: bool, rvb: bool, dwr: bool, tag: str) -> str:
    srd_head = rvb or dwr
    p4_cat, pose_idx = (12, [15, 18, 21]) if srd_head else (13, [16, 19, 22])
    bb = [
        "backbone:",
        "  - [-1, 1, Conv, [64, 3, 2]]",
        "  - [-1, 1, Conv, [128, 3, 2]]",
        _c3(rvb, dwr, "rvb", 256, False, 0.25),
        "  - [-1, 1, Conv, [256, 3, 2]]",
        _c3(rvb, dwr, "rvb", 512, False, 0.25),
        "  - [-1, 1, Conv, [512, 3, 2]]",
        _c3(rvb, dwr, "dwr", 512, True),
        "  - [-1, 1, Conv, [1024, 3, 2]]",
        _c3(rvb, dwr, "dwr", 1024, True),
        "  - [-1, 1, SPPF, [1024, 5]]",
        "  - [-1, 2, C2PSA, [1024]]",
    ]
    # Head topology matches full SRD (concat refs 6, 4, 12, 10; P3/P4/P5 at 15/18/21).
    hd = [
        "head:",
        "  - [-1, 1, nn.Upsample, [None, 2, \"nearest\"]]",
        "  - [[-1, 6], 1, Concat, [1]]",
        _c3(rvb, dwr, "p4", 512),
        "  - [-1, 1, nn.Upsample, [None, 2, \"nearest\"]]",
        "  - [[-1, 4], 1, Concat, [1]]",
        _c3(rvb, dwr, "p3", 256),
        "  - [-1, 1, Conv, [256, 3, 2]]",
        f"  - [[-1, {p4_cat}], 1, Concat, [1]]",
        _c3(rvb, dwr, "p4d", 512),
        "  - [-1, 1, Conv, [512, 3, 2]]",
        "  - [[-1, 10], 1, Concat, [1]]",
        _c3(rvb, dwr, "p5", 1024),
    ]
    pose = "Pose_SEAM" if seam else "Pose"
    hd.append(f"  - [{pose_idx}, 1, {pose}, [nc, kpt_shape]]")
    body = HEADER.format(tag=tag, seam=seam, rvb=rvb, dwr=dwr) + "\n".join(bb) + "\n\n" + "\n".join(hd) + "\n"
    return body


ABLATIONS: dict[str, dict] = {
    "A": {"seam": False, "rvb": False, "dwr": False, "dat": False},
    "B": {"seam": False, "rvb": False, "dwr": False, "dat": True},
    "C": {"seam": True, "rvb": False, "dwr": False, "dat": True},
    "D": {"seam": False, "rvb": True, "dwr": False, "dat": True},
    "E": {"seam": False, "rvb": False, "dwr": True, "dat": True},
    "F": {"seam": True, "rvb": True, "dwr": False, "dat": True},
    "G": {"seam": True, "rvb": False, "dwr": True, "dat": True},
    "H": {"seam": False, "rvb": True, "dwr": True, "dat": True},
    "I": {"seam": True, "rvb": True, "dwr": True, "dat": True},
}

# Existing completed runs map to ablation rows (same architecture + Dat).
EXISTING_RUNS: dict[str, str] = {
    "B": "yolo11m_pose_baseline_e150_pat10_imgsz1280",
    "I": "yolo11m_pose_srd_e150_pat10_imgsz1280",
}


def run_name(letter: str) -> str:
    return f"yolo11m_srd_abl_{letter.lower()}_e150_pat10_imgsz1280"


def yaml_path(letter: str) -> Path:
    return OUT / f"yolo11m-pose-srd-abl-{letter.lower()}.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for letter, spec in ABLATIONS.items():
        path = yaml_path(letter)
        text = build_yaml(spec["seam"], spec["rvb"], spec["dwr"], letter)
        path.write_text(text)
        print(f"Wrote {path.name}  SEAM={spec['seam']} RVB={spec['rvb']} DWR={spec['dwr']} Dat={spec['dat']}")

    if args.verify:
        import sys

        ROOT = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(ROOT))
        from pdca_yolo11.srd_modules import register_srd_modules

        register_srd_modules()
        from ultralytics import YOLO

        print("\n--- verify ---")
        for letter in ABLATIONS:
            m = YOLO(str(yaml_path(letter)))
            n = sum(p.numel() for p in m.model.parameters())
            print(f"  {letter}: params={n:,} layers={len(m.model.model)}")


if __name__ == "__main__":
    main()
