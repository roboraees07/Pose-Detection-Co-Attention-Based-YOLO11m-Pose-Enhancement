#!/usr/bin/env bash
# Module ladder: Dat OFF/ON, ECA@B2, CBAM@B4, C2PSA@B10, PDCA, SEAM combinations.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python scripts/train_experiment.py --group ladder_ablation --device "${DEVICE:-0}"
