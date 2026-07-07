# ZJDKY Project Memory

This file stores persistent project requirements and working agreements.
Keep it updated whenever the project-level assumptions change.

## Persistent Requirements

- Deep learning related code for the tension clamp region recognition task must be placed under `ZJDKY/DL/codex`.
- Test-version web pages must be placed under `html`.
- Training data is stored at `D:\浙江电科院数据集\PNG标注`.
- Training data uses `.png` images with corresponding `.json` annotation files.
- The Python interpreter environment is temporarily set to the conda environment named `Pytorch`.
- Region detector training must use GPU/CUDA and must not fall back to CPU.

## Maintenance Notes

- Current first-stage goal: identify tension clamp regions 1, 2, and 3 from existing annotated data.
- First-stage implementation route: use torchvision Faster R-CNN object detection to output bounding boxes for regions 1, 2, and 3; no pixel-level segmentation is required.
- Fixed handoff context for the first-stage region detector is stored in `ZJDKY/DL/codex/REGION_DETECTOR_CONTEXT.md`.
- Second-stage goal: train defect recognition models using abnormal data from each region.
- Dataset characteristic: normal samples are abundant, abnormal samples are relatively scarce.
- Treat this file as the source of truth for durable project conventions.
- When new persistent requirements are provided, append or update them here.
