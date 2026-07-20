"""Experiment registry for all paper tables and ablation studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pdca_yolo11.constants import IMGSZ, EPOCHS, PATIENCE
from pdca_yolo11.paths import CONFIGS_DIR

RUN_TAG = f"e{EPOCHS}_pat{PATIENCE}_imgsz{IMGSZ}"


@dataclass
class Experiment:
    """One trainable/evaluable experiment."""

    id: str
    display_name: str
    run_name: str
    model: str  # ultralytics pretrained name or yaml path
    group: str
    # Module flags for paper tables
    dat: bool = True
    eca_b2: bool = False
    cbam_b4: bool = False
    c2psa_b10: bool = False
    pdca: bool = False
    seam: bool = False
    rvb: bool = False
    dwr: bool = False
    register: Literal["none", "attention", "srd", "both"] = "none"
    no_augmentation: bool = False
    yolo_family: str = "yolo11"
    scale: str = "m"
    notes: str = ""


def _run(prefix: str) -> str:
    return f"{prefix}_{RUN_TAG}"


def _yaml(*parts: str) -> str:
    return str(CONFIGS_DIR.joinpath(*parts))


# ---------------------------------------------------------------------------
# 1) Scale baselines — YOLO11n/s/m/l/x and YOLOv8n/s/m/l/x
# ---------------------------------------------------------------------------
_YOLO11_SCALES = ("n", "s", "m", "l", "x")
_YOLO8_SCALES = ("n", "s", "m", "l", "x")

SCALE_BASELINES: list[Experiment] = []
for scale in _YOLO11_SCALES:
    SCALE_BASELINES.append(
        Experiment(
            id=f"baseline_yolo11{scale}",
            display_name=f"YOLO11{scale.upper()}-Pose",
            run_name=_run(f"yolo11{scale}_pose_baseline"),
            model=f"yolo11{scale}-pose.pt",
            group="scale_baselines",
            yolo_family="yolo11",
            scale=scale,
        )
    )
for scale in _YOLO8_SCALES:
    SCALE_BASELINES.append(
        Experiment(
            id=f"baseline_yolov8{scale}",
            display_name=f"YOLOv8{scale.upper()}-Pose",
            run_name=_run(f"yolov8{scale}_pose_baseline"),
            model=f"yolov8{scale}-pose.pt",
            group="scale_baselines",
            yolo_family="yolov8",
            scale=scale,
        )
    )


# ---------------------------------------------------------------------------
# 2) Module ladder ablation (ECA@B2, CBAM@B4, C2PSA@B10, PDCA, SEAM)
# ---------------------------------------------------------------------------
LADDER_ABLATION: list[Experiment] = [
    Experiment(
        id="ladder_no_aug",
        display_name="YOLO11m-Pose (Dat OFF)",
        run_name=_run("yolo11m_srd_abl_a"),
        model=_yaml("srd", "ablation", "yolo11m-pose-srd-abl-a.yaml"),
        group="ladder_ablation",
        dat=False,
        no_augmentation=True,
        register="srd",
    ),
    Experiment(
        id="ladder_baseline",
        display_name="YOLO11m-Pose (Dat ON)",
        run_name=_run("yolo11m_pose_baseline"),
        model="yolo11m-pose.pt",
        group="ladder_ablation",
        dat=True,
    ),
    Experiment(
        id="ladder_eca_b2",
        display_name="YOLO11m-Pose + ECA @ B2",
        run_name=_run("yolo11m_pose_b2_eca"),
        model=_yaml("yolo11", "yolo11m-pose-attn-b2-eca.yaml"),
        group="ladder_ablation",
        dat=True,
        eca_b2=True,
        register="attention",
    ),
    Experiment(
        id="ladder_eca_cbam",
        display_name="YOLO11m-Pose + ECA @ B2 + CBAM @ B4",
        run_name=_run("yolo11m_pose_eca_cbam"),
        model=_yaml("models", "yolo11", "yolo11m-pose-eca-cbam.yaml"),
        group="ladder_ablation",
        dat=True,
        eca_b2=True,
        cbam_b4=True,
        register="attention",
    ),
    Experiment(
        id="ladder_eca_cbam_c2psa",
        display_name="YOLO11m-Pose + ECA @ B2 + CBAM @ B4 + C2PSA @ B10",
        run_name=_run("yolo11m_pose_eca_cbam_c2psa"),
        model=_yaml("models", "yolo11", "yolo11m-pose-eca-cbam-c2psa.yaml"),
        group="ladder_ablation",
        dat=True,
        eca_b2=True,
        cbam_b4=True,
        c2psa_b10=True,
        register="attention",
    ),
    Experiment(
        id="ladder_pdca_only",
        display_name="YOLO11m-Pose + PDCA",
        run_name=_run("yolo11m_pose_pdca"),
        model=_yaml("models", "yolo11", "yolo11m-pose-pdca.yaml"),
        group="ladder_ablation",
        dat=True,
        pdca=True,
        register="attention",
    ),
    Experiment(
        id="ladder_seam_only",
        display_name="YOLO11m-Pose + SEAM",
        run_name=_run("yolo11m_srd_abl_c"),
        model=_yaml("srd", "ablation", "yolo11m-pose-srd-abl-c.yaml"),
        group="ladder_ablation",
        dat=True,
        seam=True,
        register="srd",
    ),
    Experiment(
        id="ladder_eca_cbam_c2psa_pdca",
        display_name="YOLO11m-Pose + ECA @ B2 + CBAM @ B4 + C2PSA @ B10 + PDCA",
        run_name=_run("yolo11m_pose_eca_cbam_c2psa_pdca"),
        model=_yaml("models", "yolo11", "yolo11m-pose-eca-cbam-c2psa-pdca.yaml"),
        group="ladder_ablation",
        dat=True,
        eca_b2=True,
        cbam_b4=True,
        c2psa_b10=True,
        pdca=True,
        register="attention",
    ),
    Experiment(
        id="ladder_eca_cbam_c2psa_pdca_seam",
        display_name="YOLO11m-Pose + ECA @ B2 + CBAM @ B4 + C2PSA @ B10 + PDCA + SEAM",
        run_name=_run("yolo11m_pose_eca_cbam_c2psa_pdca_seam"),
        model=_yaml("models", "yolo11", "yolo11m-pose-eca-cbam-c2psa-pdca-seam.yaml"),
        group="ladder_ablation",
        dat=True,
        eca_b2=True,
        cbam_b4=True,
        c2psa_b10=True,
        pdca=True,
        seam=True,
        register="both",
    ),
]


# ---------------------------------------------------------------------------
# 3) SRD-inspired ablation (SEAM × RVB × DWR × Dat) — rows A–I
# ---------------------------------------------------------------------------
_SRD_SPECS = {
    "A": dict(seam=False, rvb=False, dwr=False, dat=False),
    "B": dict(seam=False, rvb=False, dwr=False, dat=True),
    "C": dict(seam=True, rvb=False, dwr=False, dat=True),
    "D": dict(seam=False, rvb=True, dwr=False, dat=True),
    "E": dict(seam=False, rvb=False, dwr=True, dat=True),
    "F": dict(seam=True, rvb=True, dwr=False, dat=True),
    "G": dict(seam=True, rvb=False, dwr=True, dat=True),
    "H": dict(seam=False, rvb=True, dwr=True, dat=True),
    "I": dict(seam=True, rvb=True, dwr=True, dat=True),
}

SRD_ABLATION: list[Experiment] = []
for letter, flags in _SRD_SPECS.items():
    SRD_ABLATION.append(
        Experiment(
            id=f"srd_{letter.lower()}",
            display_name=f"SRD ablation {letter}",
            run_name=_run(f"yolo11m_srd_abl_{letter.lower()}"),
            model=_yaml("srd", "ablation", f"yolo11m-pose-srd-abl-{letter.lower()}.yaml"),
            group="srd_ablation",
            register="srd",
            no_augmentation=not flags["dat"],
            **flags,
        )
    )


# ---------------------------------------------------------------------------
# 4) Method comparison — YOLO11m and YOLOv8m (8 methods)
# ---------------------------------------------------------------------------
METHOD_COMPARISON: list[Experiment] = [
    Experiment(
        id="method_yolo11m_baseline",
        display_name="YOLO11m-Pose",
        run_name=_run("yolo11m_pose_baseline"),
        model="yolo11m-pose.pt",
        group="method_comparison",
        yolo_family="yolo11",
        scale="m",
    ),
    Experiment(
        id="method_yolo11m_eca_cbam_c2psa",
        display_name="YOLO11m ECA@B2 + CBAM@B4 + C2PSA@B10",
        run_name=_run("yolo11m_pose_eca_cbam_c2psa"),
        model=_yaml("models", "yolo11", "yolo11m-pose-eca-cbam-c2psa.yaml"),
        group="method_comparison",
        eca_b2=True,
        cbam_b4=True,
        c2psa_b10=True,
        register="attention",
        yolo_family="yolo11",
        scale="m",
    ),
    Experiment(
        id="method_yolo11m_pdca",
        display_name="YOLO11m PDCA only",
        run_name=_run("yolo11m_pose_pdca"),
        model=_yaml("models", "yolo11", "yolo11m-pose-pdca.yaml"),
        group="method_comparison",
        pdca=True,
        register="attention",
        yolo_family="yolo11",
        scale="m",
    ),
    Experiment(
        id="method_yolo11m_eca_cbam_c2psa_pdca",
        display_name="YOLO11m ECA@B2 + CBAM@B4 + C2PSA@B10 + PDCA",
        run_name=_run("yolo11m_pose_eca_cbam_c2psa_pdca"),
        model=_yaml("models", "yolo11", "yolo11m-pose-eca-cbam-c2psa-pdca.yaml"),
        group="method_comparison",
        eca_b2=True,
        cbam_b4=True,
        c2psa_b10=True,
        pdca=True,
        register="attention",
        yolo_family="yolo11",
        scale="m",
    ),
    Experiment(
        id="method_yolov8m_baseline",
        display_name="YOLOv8m-Pose",
        run_name=_run("yolov8m_pose_baseline"),
        model="yolov8m-pose.pt",
        group="method_comparison",
        yolo_family="yolov8",
        scale="m",
    ),
    Experiment(
        id="method_yolov8m_eca_cbam_c2psa",
        display_name="YOLOv8m ECA@B2 + CBAM@B4 + C2PSA@B10",
        run_name=_run("yolov8m_pose_eca_cbam_c2psa"),
        model=_yaml("models", "yolo8", "yolo8m-pose-eca-cbam-c2psa.yaml"),
        group="method_comparison",
        eca_b2=True,
        cbam_b4=True,
        c2psa_b10=True,
        register="attention",
        yolo_family="yolov8",
        scale="m",
    ),
    Experiment(
        id="method_yolov8m_pdca",
        display_name="YOLOv8m PDCA only",
        run_name=_run("yolov8m_pose_pdca"),
        model=_yaml("models", "yolo8", "yolo8m-pose-pdca.yaml"),
        group="method_comparison",
        pdca=True,
        register="attention",
        yolo_family="yolov8",
        scale="m",
    ),
    Experiment(
        id="method_yolov8m_eca_cbam_c2psa_pdca",
        display_name="YOLOv8m ECA@B2 + CBAM@B4 + C2PSA@B10 + PDCA",
        run_name=_run("yolov8m_pose_eca_cbam_c2psa_pdca"),
        model=_yaml("models", "yolo8", "yolo8m-pose-eca-cbam-c2psa-pdca.yaml"),
        group="method_comparison",
        eca_b2=True,
        cbam_b4=True,
        c2psa_b10=True,
        pdca=True,
        register="attention",
        yolo_family="yolov8",
        scale="m",
    ),
]


# ---------------------------------------------------------------------------
# 5) Confidence ablation — uses two trained checkpoints (aug ON / OFF)
# ---------------------------------------------------------------------------
CONF_SWEEP_MODELS: list[Experiment] = [
    Experiment(
        id="conf_sweep_no_aug",
        display_name="YOLO11m-Pose (no augmentation)",
        run_name=_run("yolo11m_srd_abl_a"),
        model="yolo11m-pose.pt",
        group="conf_sweep",
        dat=False,
        no_augmentation=True,
    ),
    Experiment(
        id="conf_sweep_aug",
        display_name="YOLO11m-Pose (+ augmentation)",
        run_name=_run("yolo11m_pose_baseline"),
        model="yolo11m-pose.pt",
        group="conf_sweep",
        dat=True,
    ),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ALL_EXPERIMENTS: dict[str, Experiment] = {}
for group_list in (SCALE_BASELINES, LADDER_ABLATION, SRD_ABLATION, METHOD_COMPARISON, CONF_SWEEP_MODELS):
    for exp in group_list:
        ALL_EXPERIMENTS[exp.id] = exp

GROUPS: dict[str, list[Experiment]] = {
    "scale_baselines": SCALE_BASELINES,
    "ladder_ablation": LADDER_ABLATION,
    "srd_ablation": SRD_ABLATION,
    "method_comparison": METHOD_COMPARISON,
    "conf_sweep": CONF_SWEEP_MODELS,
}


def tick(v: bool) -> str:
    return "✓" if v else "×"


def module_flags_row(exp: Experiment) -> dict[str, str]:
    return {
        "Dat": tick(exp.dat),
        "ECA @ B2": tick(exp.eca_b2),
        "CBAM @ B4": tick(exp.cbam_b4),
        "C2PSA @ B10": tick(exp.c2psa_b10),
        "PDCA": tick(exp.pdca),
        "SEAM": tick(exp.seam),
    }


def srd_flags_row(exp: Experiment) -> dict[str, str]:
    return {
        "SEAM": tick(exp.seam),
        "RVB": tick(exp.rvb),
        "DWR": tick(exp.dwr),
        "Dat": tick(exp.dat),
    }
