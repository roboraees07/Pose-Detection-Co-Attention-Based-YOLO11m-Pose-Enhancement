"""Shared training and evaluation hyperparameters."""

from __future__ import annotations

EPOCHS = 150
PATIENCE = 10
LR_PATIENCE = 5
BATCH = 8
IMGSZ = 1280
EVAL_CONF = 0.30
EVAL_IOU = 0.7
LOC_IOU_THR = 0.1
WEED_CLASS_ID = 8

CONF_SWEEP_VALUES = (0.001, 0.25, 0.30, 0.35, 0.40, 0.50)

TRAIN_KW_BASE = dict(
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    optimizer="SGD",
    momentum=0.937,
    patience=PATIENCE,
    cos_lr=False,
    lrf=1.0,
    lr0=0.01,
    val=True,
    seed=0,
    deterministic=True,
    close_mosaic=10,
    save_period=10,
    pretrained=True,
    plots=True,
    verbose=True,
)

NO_AUG_KW = dict(
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    auto_augment=None,
    erasing=0.0,
    close_mosaic=0,
)

LR_PROTOCOL = "lr0=0.01 + ReduceLROnPlateau (p=5, f=0.5)"

METRIC_COLUMNS = [
    "mAP_kpt/%",
    "F1/%",
    "Params/M",
    "FPS",
    "vs YOLO11m",
    "Epoch stops",
    "Det P(%)",
    "Det R(%)",
    "Det F1(%)",
    "Det mAP50(%)",
    "Det mAP50-95(%)",
    "Pose P(%)",
    "Pose R(%)",
    "Pose F1(%)",
    "Pose mAP50(%)",
    "Pose mAP50-95(%)",
    "MEA(px)",
    "CAR(%)",
    "RMSE(%)",
    "time(ms/img)",
    "Parameters",
    "Layers",
    "GFLOPs",
]
