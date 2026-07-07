# ZJDKY Project Memory

This file stores persistent project requirements and working agreements.
Keep it updated whenever the project-level assumptions change.

## Persistent Requirements

- Deep learning related code for the tension clamp region recognition task must be placed under `ZJDKY/DL/codex`.
- Test-version web pages must be placed under `html`.
- Training data on this computer is stored at `F:\数据集\浙江电科院-底片标注\训练数据清洗\PNG标注`.
- Training data uses `.png` images with corresponding `.json` annotation files.
- The Python interpreter environment on this computer is the conda environment named `pytorch_env`.
- Region detector training must use GPU/CUDA and must not fall back to CPU.

## Maintenance Notes

- Current first-stage goal: identify tension clamp regions 1, 2, and 3 from existing annotated data.
- First-stage implementation route: use torchvision Faster R-CNN object detection to output bounding boxes for regions 1, 2, and 3; no pixel-level segmentation is required.
- Current official first-stage model uses Faster R-CNN with a pretrained ResNet50 backbone and geometry-based post-processing.
- Current official first-stage model is stored at `ZJDKY/DL/codex/outputs/region_detector/best.pt`.
- Fixed handoff context for the first-stage region detector is stored in `ZJDKY/DL/codex/REGION_DETECTOR_CONTEXT.md`.
- Second-stage goal: train defect recognition models using abnormal data from each region.
- Dataset characteristic: normal samples are abundant, abnormal samples are relatively scarce.
- Treat this file as the source of truth for durable project conventions.
- When new persistent requirements are provided, append or update them here.
