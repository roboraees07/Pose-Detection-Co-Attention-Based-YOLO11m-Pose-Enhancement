#!/usr/bin/env bash
# Train all YOLO11/YOLOv8 scale baselines (n/s/m/l/x).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python scripts/train_experiment.py --group scale_baselines --device "${DEVICE:-0}"
