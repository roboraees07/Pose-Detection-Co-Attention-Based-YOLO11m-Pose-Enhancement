#!/usr/bin/env bash
# Evaluate all experiment groups @ conf=0.30 and export CSV tables.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
DEVICE="${DEVICE:-0}"

for GROUP in scale_baselines ladder_ablation srd_ablation method_comparison; do
  echo "========== Evaluating $GROUP =========="
  python scripts/eval_experiment.py --group "$GROUP" --device "$DEVICE"
  python scripts/export_tables.py --group "$GROUP"
done

echo "========== Confidence ablation =========="
python scripts/conf_sweep.py --device "$DEVICE"

echo "Done. Results in results/"
