#!/usr/bin/env python3
"""Build YOLOv8m-pose YAMLs with ECA + CBAM + C2PSA and PDCA (CropsOrWeed9)."""

from __future__ import annotations

import argparse
from pathlib import Path

CODES = Path(__file__).resolve().parent

HEADER = """# Auto-generated YOLOv8m-pose combo. Steps: {steps}
nc: 9
kpt_shape: [1, 3]
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]
"""

COMBOS: list[tuple[str, tuple[int, ...], str | None]] = [
    ("1-2-4", (1, 2, 4), None),
    ("1-2-4-pdca", (1, 2, 4), "PDCA"),
    ("pdca", (), "PDCA"),
]

V8M_WIDTH = 0.75
V8M_MAX_CH = 768


def v8m_ch(c: int) -> int:
    return int(min(c, V8M_MAX_CH) * V8M_WIDTH + 0.5)


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
    bb_add(-1, 3, "C2f", [128, True], tag="b2")
    if 1 in steps:
        bb_add(-1, 1, "ECA", [v8m_ch(128)])
    bb_add(-1, 1, "Conv", [256, 3, 2])
    bb_add(-1, 6, "C2f", [256, True], tag="p3_bb")
    if 2 in steps:
        bb_add(-1, 1, "CBAM", [v8m_ch(256)])
    bb_add(-1, 1, "Conv", [512, 3, 2])
    bb_add(-1, 6, "C2f", [512, True], tag="p4_bb")
    if 3 in steps:
        bb_add(-1, 1, "ECA", [v8m_ch(512)])
    bb_add(-1, 1, "Conv", [1024, 3, 2])
    bb_add(-1, 3, "C2f", [1024, True])
    bb_add(-1, 1, "SPPF", [1024, 5], tag="b10")
    if 4 in steps:
        bb_add(-1, 3, "C2f", [1024, True])

    hd: list[list] = []
    off = len(bb)

    def g(hd_i: int) -> int:
        return off + hd_i

    p4_bb = tags["p4_bb"]
    p3_bb = tags["p3_bb"]
    b10 = tags["b10"]

    hd.append([-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    hd.append([[-1, p4_bb], 1, "Concat", [1]])
    hd.append([-1, 3, "C2f", [512]])
    n2 = g(len(hd) - 1)

    hd.append([-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    hd.append([[-1, p3_bb], 1, "Concat", [1]])
    hd.append([-1, 3, "C2f", [256]])
    p3_out = g(len(hd) - 1)

    hd.append([-1, 1, "Conv", [256, 3, 2]])
    hd.append([[-1, n2], 1, "Concat", [1]])
    hd.append([-1, 3, "C2f", [512]])
    p4_out = g(len(hd) - 1)

    hd.append([-1, 1, "Conv", [512, 3, 2]])
    hd.append([[-1, b10], 1, "Concat", [1]])
    hd.append([-1, 3, "C2f", [1024]])
    p5_out = g(len(hd) - 1)

    if extra == "PDCA":
        pdca_ch = (
            (p3_out, v8m_ch(256), "p3_out"),
            (p4_out, v8m_ch(512), "p4_out"),
            (p5_out, v8m_ch(1024), "p5_out"),
        )
        for idx, ch, tag in pdca_ch:
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
        step_tags = f"{step_tags},{extra}" if step_tags else extra
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
    args = p.parse_args()
    out_dir = CODES.parent / "configs" / "combos" / "yolo8"
    out_dir.mkdir(exist_ok=True)
    for slug, steps, extra in COMBOS:
        name = f"yolo8m-pose-combo-{slug}.yaml" if slug != "pdca" else "yolo8m-pose-pdca.yaml"
        path = out_dir / name
        write_yaml(path, steps, extra=extra)
        if args.verify:
            verify(path)
        print("Wrote", path)


if __name__ == "__main__":
    main()
