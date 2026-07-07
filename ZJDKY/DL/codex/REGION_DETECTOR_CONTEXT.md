# Region Detector Context

This file is the fixed handoff context for the tension clamp region 1/2/3
detection work. Keep it updated when implementation details or durable
decisions change.

## Goal

- First-stage task: identify and mark tension clamp `区域1`, `区域2`, and `区域3`.
- Output requirement: bounding boxes for whole regions only.
- Pixel-level segmentation is not required for this stage.
- Second-stage task, deferred: use region-specific abnormal samples for defect recognition.

## Durable Project Requirements

- Deep learning code must stay under `ZJDKY/DL/codex`.
- Test-version web pages must stay under `html`.
- Training data root: `D:\浙江电科院数据集\PNG标注`.
- Data format: `.png` image with matching LabelMe-style `.json` annotation.
- Python environment: conda environment named `Pytorch`.
- Region detector training must use GPU/CUDA and must not fall back to CPU.

## Data Status

- Cleaned dataset contains 291 PNG files and 291 JSON files.
- All 291 samples are paired by same folder and file stem.
- All 291 samples contain exactly 3 region annotations.
- Labels are `区域1`, `区域2`, `区域3`.
- Annotation shape type is polygon; polygons are converted to bounding boxes.
- Image modes observed: grayscale `L` and `RGB`; code converts inputs to RGB.
- Region orientation is mixed and normal:
  - left-to-right: 156 samples
  - right-to-left: 135 samples
- The model must learn orientation from data and must not hard-code region order by x-position.

## Implemented Route

- Chosen implementation route: object detection.
- Model: `torchvision` Faster R-CNN.
- No dependency on `ultralytics`.
- Class IDs:
  - background: 0
  - `区域1`: 1
  - `区域2`: 2
  - `区域3`: 3
- Training input: whole image.
- Training target: 3 bounding boxes per image.
- Default train/val/test split: 70% / 15% / 15%, fixed seed `20260707`.

## Implemented Files

- `region_dataset.py`
  - Reads LabelMe JSON.
  - Repairs/normalizes region labels when needed.
  - Converts polygon points to bbox.
  - Provides `RegionDetectionDataset` and `collate_fn`.
  - Writes paths in portable forward-slash form inside split files.

- `prepare_region_data.py`
  - Discovers all samples.
  - Builds `splits.json`.
  - Builds `label_map.json`.
  - Writes `prepare_stats.json`.
  - Generates ground-truth preview images.

- `train_region_detector.py`
  - Trains Faster R-CNN.
  - Requires CUDA/GPU.
  - Saves `best.pt`, `last.pt`, `metrics.json`, and `training_log.csv`.
  - Includes lightweight validation metrics: mAP@0.5 approximation and mean IoU.

- `predict_regions.py`
  - Loads trained checkpoint.
  - Supports one PNG or a directory of PNGs.
  - Writes `predictions.csv`.
  - Writes `prediction_anomalies.csv` for missing/duplicate regions.
  - Generates prediction visualizations.

## Output Locations

- Main output directory: `ZJDKY/DL/codex/outputs/region_detector`
- Data preparation outputs:
  - `splits.json`
  - `label_map.json`
  - `prepare_stats.json`
  - `visualizations/ground_truth/*.png`
- Training outputs:
  - `best.pt`
  - `last.pt`
  - `metrics.json`
  - `training_log.csv`
- Official model weights under `outputs/region_detector` may be uploaded to Git.
- Generated images, smoke-test outputs, prediction CSV/JSON files, and logs should stay ignored.
- Prediction outputs:
  - `predictions.csv`
  - `prediction_anomalies.csv`
  - `prediction_summary.json`
  - `visualizations/predictions/*.png`

## Verified Commands

Data preparation:

```powershell
conda run -n Pytorch python ZJDKY\DL\codex\prepare_region_data.py --data-root "D:\浙江电科院数据集\PNG标注"
```

Result:

- total samples: 291
- train: 203
- val: 43
- test: 45
- error count: 0

Official GPU training command:

```powershell
conda run -n Pytorch python ZJDKY\DL\codex\train_region_detector.py --data-root "D:\浙江电科院数据集\PNG标注"
```

Prediction command:

```powershell
conda run -n Pytorch python ZJDKY\DL\codex\predict_regions.py --weights ZJDKY\DL\codex\outputs\region_detector\best.pt --input <png-or-directory>
```

## Verification Already Performed

- Syntax check passed for:
  - `region_dataset.py`
  - `prepare_region_data.py`
  - `train_region_detector.py`
  - `predict_regions.py`
- Smoke training was run with a tiny 16-sample subset only to verify the pipeline.
- Smoke test used CPU before the GPU-only requirement was added; formal training must not use CPU.
- Current conda `Pytorch` environment reports CUDA available:
  - GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`

## Important Notes

- `best.pt` under `outputs/region_detector_smoke` is only a smoke-test checkpoint and is not a usable production model.
- Formal training should produce `outputs/region_detector/best.pt`.
- If training quality is insufficient with 291 samples, next options are:
  - tune input size and epochs,
  - add data augmentation,
  - freeze/unfreeze backbone in stages,
  - consider another detection framework later.
- Do not change or rewrite original training images or JSON annotations from code.
