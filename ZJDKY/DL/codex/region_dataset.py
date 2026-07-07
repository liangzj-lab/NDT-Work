"""Dataset utilities for tension clamp region detection."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image


DEFAULT_DATA_ROOT = Path("D:/\u6d59\u6c5f\u7535\u79d1\u9662\u6570\u636e\u96c6/PNG\u6807\u6ce8")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "region_detector"

LABEL_TO_ID = {
    "\u533a\u57df1": 1,
    "\u533a\u57df2": 2,
    "\u533a\u57df3": 3,
}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
BACKGROUND_ID = 0
NUM_CLASSES = 4


@dataclass(frozen=True)
class RegionSample:
    image_path: str
    json_path: str
    width: int
    height: int
    boxes: list[list[float]]
    labels: list[int]


def portable_path(path: Path) -> str:
    return path.resolve().as_posix()


def label_map_payload() -> dict[str, Any]:
    return {
        "background": BACKGROUND_ID,
        "labels": LABEL_TO_ID,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
        "num_classes": NUM_CLASSES,
    }


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def normalize_label(label: str) -> str:
    label = label.strip().replace(" ", "")
    candidates = [label]
    for source_encoding in ("gbk", "cp936"):
        try:
            candidates.append(label.encode(source_encoding).decode("utf-8"))
        except UnicodeError:
            pass
    for candidate in candidates:
        if candidate in LABEL_TO_ID:
            return candidate
    for idx in ("1", "2", "3"):
        if idx in label and ("\u533a" in label or "\u57df" in label or "\u9356" in label):
            return f"\u533a\u57df{idx}"
    return label


def find_image_for_json(json_path: Path, image_path_value: Any) -> Path:
    if isinstance(image_path_value, str) and image_path_value:
        candidate = json_path.parent / image_path_value
        if candidate.exists():
            return candidate
    for suffix in (".png", ".PNG"):
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return json_path.with_suffix(".png")


def bbox_from_points(points: Iterable[Iterable[float]]) -> list[float]:
    parsed = [(float(point[0]), float(point[1])) for point in points]
    xs = [point[0] for point in parsed]
    ys = [point[1] for point in parsed]
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_labelme_sample(json_path: Path) -> RegionSample:
    payload = read_json(json_path)
    image_path = find_image_for_json(json_path, payload.get("imagePath"))
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image for {json_path}: {image_path}")

    with Image.open(image_path) as image:
        width, height = image.size

    boxes_by_label: dict[str, list[float]] = {}
    for shape in payload.get("shapes") or []:
        label = normalize_label(str(shape.get("label", "")))
        if label not in LABEL_TO_ID:
            continue
        points = shape.get("points") or []
        box = bbox_from_points(points)
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox for {json_path}: {label} {box}")
        boxes_by_label[label] = [x1, y1, x2, y2]

    missing = [label for label in LABEL_TO_ID if label not in boxes_by_label]
    if missing:
        raise ValueError(f"Missing region labels in {json_path}: {missing}")

    labels = sorted(LABEL_TO_ID.values())
    boxes = [boxes_by_label[ID_TO_LABEL[label_id]] for label_id in labels]
    return RegionSample(
        image_path=portable_path(image_path),
        json_path=portable_path(json_path),
        width=width,
        height=height,
        boxes=boxes,
        labels=labels,
    )


def discover_samples(data_root: Path) -> list[RegionSample]:
    samples: list[RegionSample] = []
    for json_path in sorted(data_root.rglob("*.json")):
        samples.append(parse_labelme_sample(json_path))
    return samples


def make_splits(
    samples: list[RegionSample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 20260707,
    limit: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    selected = list(samples)
    rng = random.Random(seed)
    rng.shuffle(selected)
    if limit is not None:
        selected = selected[:limit]

    n_total = len(selected)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    if n_total >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
    n_test = n_total - n_train - n_val
    if n_total >= 3 and n_test <= 0:
        n_train = max(1, n_train - 1)

    split_samples = {
        "train": selected[:n_train],
        "val": selected[n_train:n_train + n_val],
        "test": selected[n_train + n_val:],
    }
    return {
        key: [sample.__dict__ for sample in value]
        for key, value in split_samples.items()
    }


def load_splits(path: Path) -> dict[str, list[RegionSample]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: [RegionSample(**sample) for sample in value]
        for key, value in payload.items()
    }


class RegionDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[RegionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor = pil_to_float_tensor(image)
        boxes = torch.as_tensor(sample.boxes, dtype=torch.float32)
        labels = torch.as_tensor(sample.labels, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(sample.labels),), dtype=torch.int64),
        }
        return image_tensor, target


def pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    import numpy as np

    array = np.asarray(image, dtype="float32") / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape[2] == 1:
        array = array.repeat(3, axis=2)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    images, targets = zip(*batch)
    return list(images), list(targets)
