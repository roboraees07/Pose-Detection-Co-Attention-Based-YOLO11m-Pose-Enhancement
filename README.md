# YoloPose-PDCA

**Pose Detection Co-Attention Based YOLO11m-Pose Enhancement for Correct Assignment Rate and Growing Point Localization**

This repository contains training and evaluation code for weed detection and growing-point localization on the **CropsOrWeed9** dataset. The proposed model extends **YOLO11m-Pose** with **ECA**, **CBAM**, an extra **C2PSA** block, and **PDCA** (Pose–Detection Co-Attention).

| Item | Location |
|------|----------|
| Proposed architecture | `configs/models/yolo11/yolo11m-pose-eca-cbam-c2psa-pdca.yaml` |
| PDCA module | `pdca_yolo11/attention_modules.py` |
| Dataset prep script | `scripts/prepare_cropsorweed9.py` |

---

## Architecture

### Complete pipeline

The full model processes field images through preprocessing, a YOLO11 backbone with attention blocks, a PAN/FPN neck, three PDCA modules on P3/P4/P5, and a joint Pose head that outputs bounding boxes and the weed growing-point keypoint.

![PDCA-YOLO11-Pose complete pipeline](docs/figures/full_pipeline.png)

### Attention modules

**ECA (Efficient Channel Attention)** recalibrates channel weights on early backbone features. This helps thin stems and edges remain visible before deeper downsampling.

![ECA module](docs/figures/eca.png)

**CBAM (Convolutional Block Attention Module)** applies channel and spatial attention on mid-level features, emphasizing regions that contain plant structure.

![CBAM module](docs/figures/cbam.png)

**C2PSA** adds position-sensitive self-attention on deep semantic features, improving context for crowded or overlapping plants.

![C2PSA module](docs/figures/c2psa.png)

**PDCA (Pose–Detection Co-Attention)** is the proposed module. It splits neck features into detection-oriented and pose-oriented paths, cross-gates them, and fuses the result before the Pose head. Pose features guide the detection gate and detection features guide the pose gate, so growing-point localization benefits from box context and vice versa.

![PDCA module](docs/figures/pdca.png)

---

## Dataset: CropAndWeed → CropsOrWeed9

### Source

We use the **CropAndWeed** dataset (WACV 2023):

- **Repository:** [https://github.com/cropandweed/cropandweed-dataset](https://github.com/cropandweed/cropandweed-dataset)
- **Paper:** Steininger et al., *The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation*, WACV 2023

Download the raw dataset from the repository above, then follow their setup instructions to generate the **CropsOrWeed9** bounding-box variant.

### Post-processing (raw CropAndWeed → YOLO-Pose)

The raw CropAndWeed release provides CSV bounding-box annotations with stem coordinates. We convert this into YOLO-Pose format with `scripts/prepare_cropsorweed9.py`. The script performs the following steps:

1. **Class mapping** — maps CropAndWeed source labels to 9 merged classes (maize stages, crops, and all weeds → class 8) using the official `CropsOrWeed9` mapping from the CropAndWeed utilities.
2. **BBox + keypoint conversion** — for each instance, reads `(left, top, right, bottom, class_id, stem_x, stem_y)` from CSV and writes YOLO-Pose lines: `class cx cy w h kx ky v` with normalized coordinates.
3. **Stem visibility** — if the stem point lies inside the bounding box, visibility `v=2`; otherwise `v=0`.
4. **Invalid filtering** — skips boxes smaller than 1 pixel or out-of-range class IDs.
5. **Train/val/test split** — 70% / 15% / 15% with random seed 42 (saved in `data/data_split_by_variant_pose.json`).
6. **Folder layout** — writes `images/{train,val,test}/` and `labels/{train,val,test}/` under `data/CropsOrWeed9/`.
7. **Dataset YAML** — updates `configs/cropsorweed9_yolopose.yaml` for Ultralytics training.

```bash
# After downloading CropAndWeed and running their CropsOrWeed9 mapping:
python scripts/prepare_cropsorweed9.py \
  --cnw-data /path/to/cropandweed-dataset/data \
  --cnw-repo /path/to/cropandweed-dataset \
  --out data/CropsOrWeed9 \
  --copy-images
```

The preprocessed dataset is **not** included in this repository (~11 GB). See [data/README.md](data/README.md) for class list and label format.

**Weed class:** ID **8** (all weed species merged). All weed-only metrics in our experiments use this class.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

Place the prepared dataset at `data/CropsOrWeed9/` before training.

---

## Train

```bash
# Augmented YOLO11m baseline
python scripts/train_experiment.py --id ladder_baseline

# Proposed model: ECA + CBAM + C2PSA + PDCA
python scripts/train_experiment.py --id ladder_eca_cbam_c2psa_pdca
```

List all experiment IDs:

```bash
python scripts/list_experiments.py
```

### Main experiment IDs

| Model | Experiment ID |
|-------|---------------|
| YOLO11m baseline (+ augmentation) | `ladder_baseline` |
| + ECA | `ladder_eca_b2` |
| + ECA + CBAM | `ladder_eca_cbam` |
| + ECA + CBAM + C2PSA | `ladder_eca_cbam_c2psa` |
| + PDCA only | `ladder_pdca_only` |
| **+ ECA + CBAM + C2PSA + PDCA (proposed)** | `ladder_eca_cbam_c2psa_pdca` |

---

## Evaluate

```bash
python scripts/eval_experiment.py --id ladder_eca_cbam_c2psa_pdca
python scripts/export_tables.py --group ladder_ablation
```

Metrics are reported for **weed (class 8)** and **overall (9-class macro)** at confidence 0.30.

---

## Reproduce experiments

```bash
bash scripts/run_ladder_ablation.sh
bash scripts/run_method_comparison.sh
bash scripts/run_srd_ablation.sh
bash scripts/run_scale_baselines.sh
bash scripts/run_eval_all.sh
```

---

## Training settings

| Setting | Value |
|---------|-------|
| Image size | 1280 |
| Batch | 8 |
| Epochs | 150 (early stop patience 10) |
| Optimizer | SGD, lr0=0.01 |
| LR schedule | ReduceLROnPlateau (patience 5, factor 0.5) |
| Eval confidence | 0.30 |

---

## Results

Pre-computed tables: `results/reference/`

---

## Citation

```bibtex
@InProceedings{Steininger_2023_WACV,
    author    = {Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena},
    title     = {The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation},
    booktitle = {WACV},
    year      = {2023}
}
```

Please also cite your paper if you use this code.
