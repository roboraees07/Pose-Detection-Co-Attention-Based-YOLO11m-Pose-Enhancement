#!/usr/bin/env bash
# Full reproduction pipeline: train all groups, then evaluate and export tables.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

bash scripts/run_scale_baselines.sh
bash scripts/run_ladder_ablation.sh
bash scripts/run_srd_ablation.sh
bash scripts/run_method_comparison.sh
bash scripts/run_eval_all.sh
