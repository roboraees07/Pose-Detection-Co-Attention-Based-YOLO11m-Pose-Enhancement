# CropsOrWeed9 Dataset

This project uses the **CropsOrWeed9** variant of the [CropAndWeed dataset](https://github.com/cropandweed/cropandweed-dataset) (WACV 2023), converted to **YOLO-Pose** format for growing-point localization.

## Download the raw dataset

1. Clone or download: [https://github.com/cropandweed/cropandweed-dataset](https://github.com/cropandweed/cropandweed-dataset)
2. Follow their README to download images and annotations.
3. Run their mapping script for the CropsOrWeed9 variant:
   ```bash
   cd cropandweed-dataset
   python cnw/map_dataset.py --dataset_target CropsOrWeed9
   ```

## Convert to YOLO-Pose (our post-processing)

```bash
python scripts/prepare_cropsorweed9.py \
  --cnw-data /path/to/cropandweed-dataset/data \
  --cnw-repo /path/to/cropandweed-dataset \
  --out data/CropsOrWeed9 \
  --copy-images
```

### What the script does

| Step | Description |
|------|-------------|
| Class merge | 9 classes via official CropsOrWeed9 mapping (all weeds → class 8) |
| Label format | CSV bbox + stem → YOLO-Pose `class cx cy w h kx ky v` |
| Normalization | All coordinates scaled to [0, 1] |
| Stem visibility | `v=2` if stem inside bbox, else `v=0` |
| Split | 70% train / 15% val / 15% test, seed 42 |
| Output | `data/CropsOrWeed9/images/` and `labels/` |

## Local folder layout

```
data/CropsOrWeed9/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── dataset_info.json
```

**Size:** ~11 GB (not stored in Git — too large for GitHub).

## Classes (9)

| ID | Name | Notes |
|----|------|-------|
| 0 | Maize | All maize growth stages merged |
| 1 | Sugar beet | |
| 2 | Soy | |
| 3 | Sunflower | |
| 4 | Potato | |
| 5 | Pea | |
| 6 | Bean | Common + faba bean |
| 7 | Pumpkin | |
| 8 | **Weed** | All weed species merged |

**Weed-only metrics** use class **8**.

## Split file

`data/data_split_by_variant_pose.json` — image stems per split (seed 42).

## Citation

```bibtex
@InProceedings{Steininger_2023_WACV,
    author    = {Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena},
    title     = {The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation},
    booktitle = {WACV},
    year      = {2023}
}
```
