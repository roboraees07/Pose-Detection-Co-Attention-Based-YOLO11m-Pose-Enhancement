#!/usr/bin/env bash
# Method comparison: YOLO11m and YOLOv8m (baseline, ECA+CBAM+C2PSA, PDCA, full model).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python scripts/train_experiment.py --group method_comparison --device "${DEVICE:-0}"
