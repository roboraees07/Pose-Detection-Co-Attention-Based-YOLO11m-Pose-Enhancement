"""Training helpers shared by all experiment scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

from pdca_yolo11.attention_modules import register_attention_modules
from pdca_yolo11.callbacks import register_reduce_lr_callback
from pdca_yolo11.constants import BATCH, IMGSZ, LR_PATIENCE, NO_AUG_KW, PATIENCE, TRAIN_KW_BASE
from pdca_yolo11.experiments import Experiment
from pdca_yolo11.paths import DATA_YAML, RUNS_DIR
from pdca_yolo11.srd_modules import register_srd_modules


def ensure_imports() -> None:
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def register_modules(exp: Experiment) -> None:
    if exp.register in ("attention", "both"):
        register_attention_modules()
    if exp.register in ("srd", "both"):
        register_srd_modules()


def pick_device(requested: str | int) -> str | int:
    if str(requested).lower() == "cpu":
        return "cpu"
    idx = int(requested)
    if not torch.cuda.is_available():
        print("CUDA unavailable -> CPU")
        return "cpu"
    torch.cuda.set_device(idx)
    _ = torch.empty(1, device=f"cuda:{idx}")
    print(f"Using GPU {idx}: {torch.cuda.get_device_name(idx)}")
    return idx


def can_resume(ckpt: Path) -> bool:
    if not ckpt.is_file():
        return False
    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        return "epoch" in ck and "optimizer" in ck
    except Exception:
        return False


def train_experiment(
    exp: Experiment,
    *,
    device: str | int = "0",
    resume: bool = False,
    batch: int = BATCH,
    epochs: int = TRAIN_KW_BASE["epochs"],
    patience: int = PATIENCE,
    lr_patience: int = LR_PATIENCE,
) -> Path:
    """Train one experiment; returns path to best.pt."""
    register_modules(exp)
    run_dir = RUNS_DIR / exp.run_name
    ckpt = run_dir / "weights" / "last.pt"
    do_resume = resume and can_resume(ckpt)
    init = str(ckpt if do_resume else exp.model)

    model = YOLO(init)
    register_reduce_lr_callback(model, patience=lr_patience, factor=0.5, min_lr=1e-6)

    train_kw = dict(
        TRAIN_KW_BASE,
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=batch,
        device=pick_device(device),
        project=str(RUNS_DIR),
        name=exp.run_name,
        exist_ok=True,
        patience=patience,
        resume=do_resume,
    )
    if exp.no_augmentation:
        train_kw.update(NO_AUG_KW)

    print(f"Training: {exp.id} — {exp.display_name}")
    print(f"  Model init: {init}")
    print(f"  Run dir:    {run_dir}")
    model.train(**train_kw)
    best = run_dir / "weights" / "best.pt"
    print(f"Done. Best weights: {best}")
    return best
