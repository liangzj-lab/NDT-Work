"""Prepare train/val/test splits for region detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from region_dataset import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    ID_TO_LABEL,
    discover_samples,
    label_map_payload,
    make_splits,
)


COLORS = {
    1: (46, 204, 113),
    2: (52, 152, 219),
    3: (231, 76, 60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview-count", type=int, default=24)
    return parser.parse_args()


def draw_sample(sample: dict, out_path: Path) -> None:
    image = Image.open(sample["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    for box, label_id in zip(sample["boxes"], sample["labels"]):
        color = COLORS[int(label_id)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        label = ID_TO_LABEL[int(label_id)]
        draw.text((x1 + 6, y1 + 6), label, fill=color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def validate_splits(splits: dict[str, list[dict]]) -> dict:
    errors: list[str] = []
    total_samples = 0
    split_counts = {}
    for split_name, samples in splits.items():
        split_counts[split_name] = len(samples)
        total_samples += len(samples)
        for sample in samples:
            if len(sample["boxes"]) != 3 or len(sample["labels"]) != 3:
                errors.append(f"{sample['json_path']}: expected 3 boxes and labels")
            width = float(sample["width"])
            height = float(sample["height"])
            for box in sample["boxes"]:
                x1, y1, x2, y2 = [float(value) for value in box]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    errors.append(f"{sample['json_path']}: bbox out of bounds {box}")
    return {
        "total_samples": total_samples,
        "split_counts": split_counts,
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(args.data_root)
    splits = make_splits(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit=args.limit,
    )
    validation = validate_splits(splits)
    if validation["error_count"]:
        raise RuntimeError(json.dumps(validation, ensure_ascii=False, indent=2))

    split_path = args.output_dir / "splits.json"
    label_map_path = args.output_dir / "label_map.json"
    stats_path = args.output_dir / "prepare_stats.json"
    split_path.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    label_map_path.write_text(
        json.dumps(label_map_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_dir = args.output_dir / "visualizations" / "ground_truth"
    preview_samples = (splits["train"] + splits["val"] + splits["test"])[:args.preview_count]
    for idx, sample in enumerate(preview_samples):
        draw_sample(sample, preview_dir / f"sample_{idx:03d}.png")

    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"Wrote {split_path}")
    print(f"Wrote {label_map_path}")
    print(f"Wrote {preview_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
