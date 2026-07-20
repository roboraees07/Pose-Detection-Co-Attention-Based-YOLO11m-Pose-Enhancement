#!/usr/bin/env python3
"""Build YOLO11m-pose YAMLs with cumulative ★#1–#7,#9–#12 attention (skip #8)."""

from __future__ import annotations

import argparse
from pathlib import Path

CODES = Path(__file__).resolve().parent

HEADER = """# Auto-generated cumulative combo (SGD fair ladder). Steps: {steps}
nc: 9
kpt_shape: [1, 3]
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
"""

# Paper IDs in cumulative order (no #8).
LADDER_IDS = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12)

LADDER_PREFIXES: list[tuple[str, tuple[int, ...]]] = [
    ("1-2", (1, 2)),
    ("1-2-3", (1, 2, 3)),
    ("1-2-3-4", (1, 2, 3, 4)),
    ("1-2-3-4-5", (1, 2, 3, 4, 5)),
    ("1-2-3-4-5-6", (1, 2, 3, 4, 5, 6)),
    ("1-2-3-4-5-6-7", (1, 2, 3, 4, 5, 6, 7)),
    ("1-2-3-4-5-6-7-9", (1, 2, 3, 4, 5, 6, 7, 9)),
    ("1-2-3-4-5-6-7-9-10", (1, 2, 3, 4, 5, 6, 7, 9, 10)),
    ("1-2-3-4-5-6-7-9-10-11", (1, 2, 3, 4, 5, 6, 7, 9, 10, 11)),
    ("1-2-3-4-5-6-7-9-10-11-12", LADDER_IDS),
]

# Selective tier experiments (non-cumulative subsets; skip #7/#8).
TIER_COMBOS: list[tuple[str, tuple[int, ...]]] = [
    # Tier 1 — extend best stack safely
    ("1-2-3-4-9", (1, 2, 3, 4, 9)),
    ("1-2-3-4-5-9", (1, 2, 3, 4, 5, 9)),
    ("1-2-3-4-12", (1, 2, 3, 4, 12)),
    ("1-2-3-4-5-6-9", (1, 2, 3, 4, 5, 6, 9)),
    # Tier 2 — lightweight
    ("2-4-5-9", (2, 4, 5, 9)),
    ("5-9", (5, 9)),
    ("1-4-5", (1, 4, 5)),
    ("4-5-9", (4, 5, 9)),
    # Tier 3 — diagnostic ablations
    ("1-2-3-5", (1, 2, 3, 5)),
    ("1-3-4", (1, 3, 4)),
    ("1-2-4", (1, 2, 4)),
    # Tier 4 — neck/head refinements
    ("1-2-3-4-10", (1, 2, 3, 4, 10)),
    ("1-2-3-4-11", (1, 2, 3, 4, 11)),
    ("9-12", (9, 12)),
]

# Best tier combos + novel modules (one module per run; do not stack).
SADA_COMBOS: list[tuple[str, tuple[int, ...]]] = [
    ("1-2-4-sada", (1, 2, 4)),
    ("1-2-3-5-sada", (1, 2, 3, 5)),
]

# (yaml slug, base steps, module name)
NOVEL_COMBOS: list[tuple[str, tuple[int, ...], str]] = [
    ("1-2-4-pdca", (1, 2, 4), "PDCA"),
    ("1-2-4-sawa", (1, 2, 4), "SAWA"),
    ("1-2-3-5-kug", (1, 2, 3, 5), "KUG"),
    ("1-2-4-lsam", (1, 2, 4), "LSAM"),
]

P3_EXTRAS = frozenset({"SADA", "SAWA", "KUG", "LSAM"})
HEAD_EXTRAS = frozenset({"PDCA"})


def _fmt(layer: list) -> str:
    f, n, m, args = layer
    return f"  - [{f}, {n}, {m}, {args}]"


def build_arch(steps: set[int], extra: str | None = None) -> tuple[list[list], list[list]]:
    bb: list[list] = []
    tags: dict[str, int] = {}

    def bb_add(f, n, m, args, tag: str | None = None) -> int:
        bb.append([f, n, m, args])
        i = len(bb) - 1
        if tag:
            tags[tag] = i
        return i

    bb_add(-1, 1, "Conv", [64, 3, 2])
    bb_add(-1, 1, "Conv", [128, 3, 2])
    bb_add(-1, 2, "C3k2", [256, False, 0.25], tag="b2")
    if 1 in steps:
        bb_add(-1, 1, "ECA", [256])
    bb_add(-1, 1, "Conv", [256, 3, 2])
    bb_add(-1, 2, "C3k2", [512, False, 0.25], tag="p3_bb")
    if 2 in steps:
        bb_add(-1, 1, "CBAM", [512])
    bb_add(-1, 1, "Conv", [512, 3, 2])
    bb_add(-1, 2, "C3k2", [512, True], tag="p4_bb")
    if 3 in steps:
        bb_add(-1, 1, "ECA", [512])
    bb_add(-1, 1, "Conv", [1024, 3, 2])
    bb_add(-1, 2, "C3k2", [1024, True])
    bb_add(-1, 1, "SPPF", [1024, 5])
    if 5 in steps:
        bb_add(-1, 3, "C2PSA", [1024], tag="b10")
    else:
        bb_add(-1, 2, "C2PSA", [1024], tag="b10")
    if 4 in steps:
        bb_add(-1, 1, "C2PSA", [1024])

    hd: list[list] = []
    off = len(bb)

    def g(hd_i: int) -> int:
        return off + hd_i

    p4_bb = tags["p4_bb"]
    p3_bb = tags["p3_bb"]
    b10 = tags["b10"]

    hd.append([-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    hd.append([[-1, p4_bb], 1, "Concat", [1]])
    hd.append([-1, 2, "C3k2", [512, False]])
    n2 = g(len(hd) - 1)
    if 6 in steps:
        hd.append([-1, 1, "ECA", [512]])
        n2 = g(len(hd) - 1)

    hd.append([-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    hd.append([[-1, p3_bb], 1, "Concat", [1]])
    if 9 in steps:
        hd.append([-1, 1, "Conv", [256, 1, 1]])
        hd.append([-1, 1, "DCN", [256]])
    hd.append([-1, 2, "C3k2", [256, False]])
    p3_out = g(len(hd) - 1)
    if extra in P3_EXTRAS:
        hd.append([-1, 1, extra, [256]])
        p3_out = g(len(hd) - 1)
    if 7 in steps:
        hd.append([-1, 1, "C2PSA", [256]])
        p3_out = g(len(hd) - 1)

    hd.append([-1, 1, "Conv", [256, 3, 2]])
    if 12 in steps:
        hd.append([-1, 1, "SE", [256]])
    hd.append([[-1, n2], 1, "Concat", [1]])
    hd.append([-1, 2, "C3k2", [512, False]])
    p4_out = g(len(hd) - 1)
    if 10 in steps:
        hd.append([-1, 1, "ECA", [512]])
        p4_out = g(len(hd) - 1)

    hd.append([-1, 1, "Conv", [512, 3, 2]])
    if 12 in steps:
        hd.append([-1, 1, "SE", [512]])
    hd.append([[-1, b10], 1, "Concat", [1]])
    hd.append([-1, 2, "C3k2", [1024, True]])
    p5_out = g(len(hd) - 1)
    if 11 in steps:
        hd.append([-1, 1, "CBAM", [512]])
        p5_out = g(len(hd) - 1)
    if 12 in steps:
        hd.append([-1, 1, "SE", [512]])
        p5_out = g(len(hd) - 1)

    if extra == "PDCA":
        for idx, ch, tag in ((p3_out, 256, "p3_out"), (p4_out, 512, "p4_out"), (p5_out, 512, "p5_out")):
            hd.append([idx, 1, "PDCA", [ch]])
            if tag == "p3_out":
                p3_out = g(len(hd) - 1)
            elif tag == "p4_out":
                p4_out = g(len(hd) - 1)
            else:
                p5_out = g(len(hd) - 1)

    hd.append([[p3_out, p4_out, p5_out], 1, "Pose", ["nc", "kpt_shape"]])
    return bb, hd


def write_yaml(path: Path, steps: tuple[int, ...], extra: str | None = None) -> None:
    s = set(steps)
    bb, hd = build_arch(s, extra=extra)
    step_tags = ",".join(f"#{x}" for x in steps)
    if extra:
        step_tags += f",{extra}"
    lines = [HEADER.format(steps=step_tags), "", "backbone:"]
    lines.extend(_fmt(x) for x in bb)
    lines.append("")
    lines.append("head:")
    lines.extend(_fmt(x) for x in hd)
    lines.append("")
    path.write_text("\n".join(lines))


def verify(path: Path) -> None:
    import sys

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    from pdca_yolo11.attention_modules import register_attention_modules

    register_attention_modules()
    from ultralytics import YOLO

    m = YOLO(str(path))
    _ = len(m.model.model)
    print(f"OK {path.name}: {len(m.model.model)} layers")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--prefix", default="")
    args = p.parse_args()
    out_dir = CODES.parent / "configs" / "combos" / "yolo11"
    out_dir.mkdir(exist_ok=True)
    all_combos = list(LADDER_PREFIXES) + list(TIER_COMBOS) + list(SADA_COMBOS)
    for slug, steps in all_combos:
        name = f"yolo11m-pose-combo-{slug}.yaml"
        path = out_dir / name
        extra = "SADA" if slug.endswith("-sada") else None
        write_yaml(path, steps, extra=extra)
        if args.verify:
            verify(path)
        print("Wrote", path)
    for slug, steps, extra in NOVEL_COMBOS:
        name = f"yolo11m-pose-combo-{slug}.yaml"
        path = out_dir / name
        write_yaml(path, steps, extra=extra)
        if args.verify:
            verify(path)
        print("Wrote", path)


if __name__ == "__main__":
    main()
