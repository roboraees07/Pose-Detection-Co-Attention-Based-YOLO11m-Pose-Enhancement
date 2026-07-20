#!/usr/bin/env bash
# SRD-inspired ablation rows A–I (SEAM × RVB × DWR × Dat).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python scripts/train_experiment.py --group srd_ablation --device "${DEVICE:-0}"
