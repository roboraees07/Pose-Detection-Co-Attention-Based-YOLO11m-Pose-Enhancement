"""Project paths — all scripts resolve paths relative to the repo root."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_YAML = CONFIGS_DIR / "cropsorweed9_yolopose.yaml"
WEED_CLASS_ID = 8

# Allow override via environment variable for custom dataset location.
DATA_ROOT = Path(os.environ.get("CROPSORWEED9_ROOT", DATA_DIR / "CropsOrWeed9"))
