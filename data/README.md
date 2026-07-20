# CropsOrWeed9 Dataset

This folder contains the **CropsOrWeed9** variant of the [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset) dataset, preprocessed for **YOLO-Pose** training.

## Included in this project

The preprocessed dataset is stored at:

```
data/CropsOrWeed9/
├── images/
│   ├── train/   (5,393 images)
│   ├── val/     (1,155 images)
│   └── test/    (1,157 images)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── all_labels/          # flat intermediate labels
└── dataset_info.json    # class list, split counts, citation
```

**Total size:** ~11 GB (images + labels)

> **GitHub note:** The dataset is listed in `.gitignore` because of size. For GitHub, host it on [Zenodo](https://zenodo.org/) or Google Drive and link in the README. The full project folder on disk includes the dataset for local reproduction.

## Label format

Each `.txt` file uses YOLO-Pose format (one instance per line):

```
class_id  cx  cy  w  h  kx  ky  v
```

All coordinates are normalized to `[0, 1]`. One keypoint per object (growing point / stem). Visibility `v`: `2` = visible, `0` = not visible.

## Classes (9)

| ID | Name        | Notes                          |
|----|-------------|--------------------------------|
| 0  | Maize       | All maize growth stages merged |
| 1  | Sugar beet  |                                |
| 2  | Soy         |                                |
| 3  | Sunflower   |                                |
| 4  | Potato      |                                |
| 5  | Pea         |                                |
| 6  | Bean        | Common + faba bean             |
| 7  | Pumpkin     |                                |
| 8  | Weed        | All weed species merged        |

**Weed-only metrics** in all experiments use class **8**.

## Train / val / test split

| Split | Images | Ratio |
|-------|--------|-------|
| train | 5,393  | 70%   |
| val   | 1,155  | 15%   |
| test  | 1,157  | 15%   |

- Random seed: **42**
- Split file: `data/data_split_by_variant_pose.json`

## Rebuild from CropAndWeed raw data

If you only have the raw CropAndWeed download:

```bash
# 1. Clone/download CropAndWeed and run their setup + mapping:
#    cd cropandweed-dataset && python cnw/map_dataset.py --dataset_target CropsOrWeed9

# 2. Build YOLO-Pose split:
python scripts/prepare_cropsorweed9.py \
  --cnw-data /path/to/cropandweed-dataset/data \
  --cnw-repo /path/to/cropandweed-dataset \
  --out data/CropsOrWeed9 \
  --copy-images
```

## Citation

```bibtex
@InProceedings{Steininger_2023_WACV,
    author    = {Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena},
    title     = {The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation},
    booktitle = {WACV},
    year      = {2023}
}
```
