"""Audit image files and region annotations.

Outputs special samples as paths relative to the PNG annotation root.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image


DEFAULT_DATA_ROOT = Path("D:/\u6d59\u6c5f\u7535\u79d1\u9662\u6570\u636e\u96c6/PNG\u6807\u6ce8")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "image_audit"
REGION_LABELS = ("\u533a\u57df1", "\u533a\u57df2", "\u533a\u57df3")


@dataclass
class SpecialSample:
    category: str
    relative_path: str
    image_relative_path: str
    json_relative_path: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def relative_to_root(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return json.loads(path.read_text(encoding=encoding)), None
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as exc:
            return None, f"json_decode_error: {exc}"
    return None, "unicode_decode_error"


def normalize_label(label: str) -> str:
    label = label.strip().replace(" ", "")
    candidates = [label]
    for source_encoding in ("gbk", "cp936"):
        try:
            candidates.append(label.encode(source_encoding).decode("utf-8"))
        except UnicodeError:
            pass
    for candidate in candidates:
        if candidate in REGION_LABELS:
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


def image_info(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return {"width": img.width, "height": img.height, "mode": img.mode}, None
    except Exception as exc:  # PIL raises several exception types for corrupt files.
        return None, f"{type(exc).__name__}: {exc}"


def bbox_from_points(points: Any) -> tuple[dict[str, float] | None, str | None]:
    if not isinstance(points, list) or len(points) < 3:
        return None, "polygon_has_fewer_than_3_points"
    parsed: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            return None, "invalid_point_format"
        try:
            parsed.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            return None, "non_numeric_point"
    xs = [point[0] for point in parsed]
    ys = [point[1] for point in parsed]
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
        "center_x": median(xs),
        "center_y": median(ys),
    }, None


def add_sample(
    samples: list[SpecialSample],
    category: str,
    root: Path,
    relative_path: Path | None,
    image_path: Path | None,
    json_path: Path | None,
    detail: str,
) -> None:
    samples.append(
        SpecialSample(
            category=category,
            relative_path=relative_to_root(relative_path, root),
            image_relative_path=relative_to_root(image_path, root),
            json_relative_path=relative_to_root(json_path, root),
            detail=detail,
        )
    )


def iqr_bounds(values: list[float], factor: float = 3.0) -> tuple[float, float] | None:
    values = sorted(values)
    if len(values) < 8:
        return None
    mid = len(values) // 2
    lower_half = values[:mid]
    upper_half = values[mid + (len(values) % 2):]
    q1 = median(lower_half)
    q3 = median(upper_half)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def write_special_csv(path: Path, rows: list[SpecialSample]) -> None:
    fieldnames = [field for field in asdict(rows[0]).keys()] if rows else [
        "category",
        "relative_path",
        "image_relative_path",
        "json_relative_path",
        "detail",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    args = parse_args()
    data_root: Path = args.data_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(
        p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() == ".png"
    )
    json_files = sorted(
        p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() == ".json"
    )
    png_by_key = {(p.parent.resolve(), p.stem): p for p in png_files}
    json_by_key = {(p.parent.resolve(), p.stem): p for p in json_files}

    specials: list[SpecialSample] = []
    image_modes: Counter[str] = Counter()
    image_sizes: Counter[str] = Counter()
    region_orientation_counts: Counter[str] = Counter()
    region_bboxes: dict[tuple[Path, str], dict[str, float]] = {}
    json_to_image: dict[Path, Path] = {}

    for png_path in png_files:
        info, error = image_info(png_path)
        if info is None:
            add_sample(
                specials,
                "unreadable_png",
                data_root,
                png_path,
                png_path,
                None,
                error or "cannot_open_image",
            )
            continue
        image_modes[str(info["mode"])] += 1
        image_sizes[f"{info['width']}x{info['height']}"] += 1

    for key, png_path in png_by_key.items():
        if key not in json_by_key:
            add_sample(
                specials,
                "png_without_json",
                data_root,
                png_path,
                png_path,
                None,
                "No JSON file with the same folder and stem.",
            )

    for key, json_path in json_by_key.items():
        if key not in png_by_key:
            add_sample(
                specials,
                "json_without_png",
                data_root,
                json_path,
                None,
                json_path,
                "No PNG file with the same folder and stem.",
            )

    for json_path in json_files:
        payload, error = read_json(json_path)
        if payload is None:
            add_sample(
                specials,
                "invalid_json",
                data_root,
                json_path,
                None,
                json_path,
                error or "cannot_parse_json",
            )
            continue

        image_path = find_image_for_json(json_path, payload.get("imagePath"))
        json_to_image[json_path] = image_path
        info, image_error = image_info(image_path) if image_path.exists() else (None, "missing")
        if info is None:
            add_sample(
                specials,
                "missing_or_unreadable_image_pair",
                data_root,
                json_path,
                image_path,
                json_path,
                image_error or "missing_or_unreadable",
            )
            continue

        json_width = payload.get("imageWidth")
        json_height = payload.get("imageHeight")
        if json_width != info["width"] or json_height != info["height"]:
            add_sample(
                specials,
                "json_image_size_mismatch",
                data_root,
                json_path,
                image_path,
                json_path,
                f"json={json_width}x{json_height}, png={info['width']}x{info['height']}",
            )

        normalized_labels: list[str] = []
        label_counts: Counter[str] = Counter()
        for shape_index, shape in enumerate(payload.get("shapes") or []):
            label = normalize_label(str(shape.get("label", "")))
            normalized_labels.append(label)
            label_counts[label] += 1
            bbox, bbox_error = bbox_from_points(shape.get("points"))
            if bbox_error:
                add_sample(
                    specials,
                    "invalid_polygon",
                    data_root,
                    json_path,
                    image_path,
                    json_path,
                    f"shape_index={shape_index}, label={label}, {bbox_error}",
                )
                continue

            assert bbox is not None
            if (
                bbox["x_min"] < 0
                or bbox["y_min"] < 0
                or bbox["x_max"] > info["width"]
                or bbox["y_max"] > info["height"]
            ):
                add_sample(
                    specials,
                    "polygon_out_of_bounds",
                    data_root,
                    json_path,
                    image_path,
                    json_path,
                    (
                        f"shape_index={shape_index}, label={label}, "
                        f"bbox=({bbox['x_min']:.1f},{bbox['y_min']:.1f},"
                        f"{bbox['x_max']:.1f},{bbox['y_max']:.1f}), "
                        f"image={info['width']}x{info['height']}"
                    ),
                )

            if label in REGION_LABELS:
                region_bboxes[(json_path, label)] = bbox

        missing = [label for label in REGION_LABELS if label not in normalized_labels]
        if missing:
            add_sample(
                specials,
                "missing_regions",
                data_root,
                json_path,
                image_path,
                json_path,
                "missing=" + "|".join(missing),
            )

        duplicates = [label for label, count in label_counts.items() if label in REGION_LABELS and count > 1]
        if duplicates:
            add_sample(
                specials,
                "duplicate_region_labels",
                data_root,
                json_path,
                image_path,
                json_path,
                "duplicates=" + "|".join(duplicates),
            )

        extra = sorted(label for label in set(normalized_labels) if label not in REGION_LABELS)
        if extra:
            add_sample(
                specials,
                "extra_labels",
                data_root,
                json_path,
                image_path,
                json_path,
                "extra=" + "|".join(extra),
            )

        if all((json_path, label) in region_bboxes for label in REGION_LABELS):
            centers = [region_bboxes[(json_path, label)]["center_x"] for label in REGION_LABELS]
            if centers[0] < centers[1] < centers[2]:
                region_orientation_counts["left_to_right"] += 1
            elif centers[0] > centers[1] > centers[2]:
                region_orientation_counts["right_to_left"] += 1
            else:
                region_orientation_counts["mixed_or_overlapping"] += 1

    metric_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (_, label), bbox in region_bboxes.items():
        for metric in ("width", "height"):
            metric_values[label][metric].append(bbox[metric])

    metric_bounds: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for label, metrics in metric_values.items():
        for metric, values in metrics.items():
            bounds = iqr_bounds(values)
            if bounds:
                metric_bounds[label][metric] = bounds

    for (json_path, label), bbox in region_bboxes.items():
        image_path = json_to_image.get(json_path)
        for metric, bounds in metric_bounds.get(label, {}).items():
            low, high = bounds
            value = bbox[metric]
            if value < low or value > high:
                add_sample(
                    specials,
                    "region_geometry_outlier",
                    data_root,
                    json_path,
                    image_path,
                    json_path,
                    f"{label} {metric}={value:.1f}, expected_range={low:.1f}..{high:.1f}",
                )

    specials = sorted(
        specials,
        key=lambda item: (item.category, item.relative_path, item.detail),
    )
    category_counts = Counter(item.category for item in specials)

    csv_path = output_dir / "special_samples.csv"
    json_path = output_dir / "image_audit.json"
    markdown_path = output_dir / "image_audit.md"
    write_special_csv(csv_path, specials)

    report = {
        "data_root": str(data_root),
        "total_png": len(png_files),
        "total_json": len(json_files),
        "image_modes": dict(image_modes),
        "image_sizes": dict(image_sizes),
        "region_orientation_counts": dict(region_orientation_counts),
        "special_sample_count": len(specials),
        "category_counts": dict(category_counts),
        "special_samples": [asdict(sample) for sample in specials],
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    category_lines = "\n".join(f"- {key}: {value}" for key, value in sorted(category_counts.items()))
    sample_lines = "\n".join(
        f"- `{sample.relative_path}` [{sample.category}] {sample.detail}"
        for sample in specials
    )
    markdown = f"""# Image Audit

## Overview

- Data root: `{data_root}`
- PNG files checked: {len(png_files)}
- JSON files checked: {len(json_files)}
- Special samples: {len(specials)}
- Region orientation counts: {dict(region_orientation_counts)}

## Special Sample Categories

{category_lines or "- None"}

## Special Samples Relative To PNG Annotation Root

{sample_lines or "- None"}

## Outputs

- CSV: `{csv_path}`
- JSON: `{json_path}`
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
