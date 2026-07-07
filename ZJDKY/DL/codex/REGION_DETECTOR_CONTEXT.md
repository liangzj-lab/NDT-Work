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
- Training data root on this computer: `F:\数据集\浙江电科院-底片标注\训练数据清洗\PNG标注`.
- Data format: `.png` image with matching LabelMe-style `.json` annotation.
- Python environment on this computer: conda environment named `pytorch_env`.
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
- Model: `torchvision` Faster R-CNN with pretrained ResNet50 backbone.
- Inference uses geometry-based post-processing: choose one candidate per label while enforcing that region 2 lies between regions 1 and 3 without hard-coding left/right orientation.
- No dependency on `ultralytics`.
- Class IDs:
  - background: 0
  - `区域1`: 1
  - `区域2`: 2
  - `区域3`: 3
- Training input: whole image.
- Training target: 3 bounding boxes per image.
- Default train/val/test split: 70% / 15% / 15%, fixed seed `20260707`.
- Official checkpoint selection uses validation `map50` first and validation `mean_iou` as the tie-breaker.

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
  - Supports `--pretrained-backbone`.
  - Saves `best.pt`, `last.pt`, `metrics.json`, and `training_log.csv`.
  - Includes lightweight validation metrics: mAP@0.5 approximation and mean IoU.

- `region_postprocess.py`
  - Provides score-based and geometry-based candidate selection.
  - Uses the structural relationship that region 2 is between regions 1 and 3.
  - Does not hard-code whether region 1 or region 3 is on the left.

- `predict_regions.py`
  - Loads trained checkpoint.
  - Supports one PNG or a directory of PNGs.
  - Defaults to geometry-based candidate selection.
  - Writes `predictions.csv`.
  - Writes `prediction_anomalies.csv` for missing/duplicate regions.
  - Generates prediction visualizations.

- `evaluate_region_detector.py`
  - Evaluates a saved split (`train`, `val`, or `test`) against ground truth.
  - Writes `summary.json`, `predictions.csv`, `prediction_anomalies.csv`, `candidates.csv`, and visualizations.

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
- Final test-set visualization output:
  - `test_predictions_best/summary.json`
  - `test_predictions_best/predictions.csv`
  - `test_predictions_best/prediction_anomalies.csv`
  - `test_predictions_best/candidates.csv`
  - `test_predictions_best/visualizations/*.png`

## Verified Commands

Data preparation:

```powershell
conda run -n pytorch_env python ZJDKY\DL\codex\prepare_region_data.py --data-root "F:\数据集\浙江电科院-底片标注\训练数据清洗\PNG标注"
```

Result:

- total samples: 291
- train: 203
- val: 43
- test: 45
- error count: 0

Official GPU training command:

```powershell
$env:TORCH_HOME="C:\Users\Administrator\Desktop\Program\python\NDT-Work\ZJDKY\DL\codex\outputs\torch_cache"
conda run -n pytorch_env python ZJDKY\DL\codex\train_region_detector.py --data-root "F:\数据集\浙江电科院-底片标注\训练数据清洗\PNG标注" --output-dir ZJDKY\DL\codex\outputs\region_detector --epochs 20 --batch-size 2 --lr 0.0025 --num-workers 0 --min-size 512 --max-size 1024 --pretrained-backbone --device cuda
```

Prediction command:

```powershell
conda run -n pytorch_env python ZJDKY\DL\codex\predict_regions.py --weights ZJDKY\DL\codex\outputs\region_detector\best.pt --input <png-or-directory>
```

Test split evaluation and visualization command:

```powershell
conda run -n pytorch_env python ZJDKY\DL\codex\evaluate_region_detector.py --weights ZJDKY\DL\codex\outputs\region_detector\best.pt --split-file ZJDKY\DL\codex\outputs\region_detector\splits.json --split test --output-dir ZJDKY\DL\codex\outputs\region_detector\test_predictions_best --score-threshold 0.50 --candidate-score-threshold 0.05 --selection geometry --iou-threshold 0.50 --device cuda
```

## Verification Already Performed

- Syntax check passed for:
  - `region_dataset.py`
  - `prepare_region_data.py`
  - `train_region_detector.py`
  - `predict_regions.py`
  - `evaluate_region_detector.py`
  - `region_postprocess.py`
- Smoke training was run with a tiny 16-sample subset only to verify the pipeline.
- Smoke test used CPU before the GPU-only requirement was added; formal training must not use CPU.
- Current conda `pytorch_env` environment reports CUDA available:
  - GPU: `NVIDIA GeForce RTX 4070 SUPER`
  - PyTorch: `2.7.1+cu128`

GPU smoke training passed on this computer:

```powershell
conda run -n pytorch_env python ZJDKY\DL\codex\train_region_detector.py --data-root "F:\数据集\浙江电科院-底片标注\训练数据清洗\PNG标注" --output-dir ZJDKY\DL\codex\outputs\region_detector_smoke_gpu --epochs 1 --batch-size 1 --limit 16 --min-size 384 --max-size 768 --num-workers 0 --device cuda
```

Smoke result:

- total samples: 16
- train: 11
- val: 2
- test: 3
- error count: 0
- epoch 1 train loss: 1.220396

Formal GPU training passed on this computer with pretrained ResNet50 backbone.
The official best checkpoint is `ZJDKY/DL/codex/outputs/region_detector/best.pt`.

Final validation result from official training:

- best map50: 1.0
- best mean IoU: 0.868993

Final test split result using official `best.pt` and geometry post-processing:

- test images: 45
- expected regions: 135
- detected regions: 135
- regions with IoU >= 0.50: 135
- region IoU@0.50 accuracy: 1.0
- mean IoU: 0.865615
- final anomaly count: 0
- per-class pass IoU@0.50:
  - region 1: 45 / 45
  - region 2: 45 / 45
  - region 3: 45 / 45

## Important Notes

- `best.pt` under `outputs/region_detector_smoke` is only a smoke-test checkpoint and is not a usable production model.
- Formal training should produce `outputs/region_detector/best.pt`.
- If training quality is insufficient with 291 samples, next options are:
  - tune input size and epochs,
  - add data augmentation,
  - freeze/unfreeze backbone in stages,
  - consider another detection framework later.
- Do not change or rewrite original training images or JSON annotations from code.
