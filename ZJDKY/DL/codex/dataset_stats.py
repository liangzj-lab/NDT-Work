"""Summarize PNG + LabelMe JSON annotations for tension clamp regions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(r"D:\浙江电科院数据集\PNG标注")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "dataset_stats"
REGION_LABELS = ("区域1", "区域2", "区域3")


@dataclass
class FileSummary:
    json_path: str
    image_path: str
    image_exists: bool
    image_width: int | None
    image_height: int | None
    shape_count: int
    labels: str
    normalized_labels: str
    has_region_1: bool
    has_region_2: bool
    has_region_3: bool
    missing_regions: str
    extra_labels: str
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with path.open("r", encoding=encoding) as f:
                return json.load(f), None
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as exc:
            return None, f"json_decode_error: {exc}"
    return None, "unicode_decode_error"


def repair_mojibake_label(label: str) -> str:
    """Repair labels such as '鍖哄煙1' back to '区域1' when possible."""
    candidates = [label]
    for source_encoding in ("gbk", "cp936"):
        try:
            candidates.append(label.encode(source_encoding).decode("utf-8"))
        except UnicodeError:
            pass

    for candidate in candidates:
        compact = candidate.replace(" ", "")
        if compact in REGION_LABELS:
            return compact

    for idx in ("1", "2", "3"):
        if idx in label:
            if "区域" in label or "鍖" in label or "煙" in label:
                return f"区域{idx}"
    return label.strip()


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def bbox_from_points(points: list[list[float]]) -> dict[str, float] | None:
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": x_max - x_min,
        "height": y_max - y_min,
        "area": polygon_area([[float(p[0]), float(p[1])] for p in points]),
    }


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


def summarize_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(sum(values) / len(values), 2),
    }


def write_csv(path: Path, rows: list[FileSummary]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
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
    png_keys = {(p.parent.resolve(), p.stem): p for p in png_files}
    json_keys = {(p.parent.resolve(), p.stem): p for p in json_files}

    raw_label_counts: Counter[str] = Counter()
    normalized_label_counts: Counter[str] = Counter()
    shape_type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    region_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    file_summaries: list[FileSummary] = []
    invalid_json: list[str] = []

    for json_path in json_files:
        payload, error = read_json(json_path)
        if payload is None:
            invalid_json.append(str(json_path))
            status_counts["invalid_json"] += 1
            file_summaries.append(
                FileSummary(
                    json_path=str(json_path),
                    image_path="",
                    image_exists=False,
                    image_width=None,
                    image_height=None,
                    shape_count=0,
                    labels="",
                    normalized_labels="",
                    has_region_1=False,
                    has_region_2=False,
                    has_region_3=False,
                    missing_regions=",".join(REGION_LABELS),
                    extra_labels="",
                    status=error or "invalid_json",
                )
            )
            continue

        image_path = find_image_for_json(json_path, payload.get("imagePath"))
        shapes = payload.get("shapes") or []
        labels: list[str] = []
        normalized_labels: list[str] = []
        for shape in shapes:
            raw_label = str(shape.get("label", "")).strip()
            normalized = repair_mojibake_label(raw_label)
            labels.append(raw_label)
            normalized_labels.append(normalized)
            raw_label_counts[raw_label] += 1
            normalized_label_counts[normalized] += 1
            shape_type_counts[str(shape.get("shape_type", ""))] += 1

            points = shape.get("points") or []
            bbox = bbox_from_points(points)
            if bbox and normalized in REGION_LABELS:
                for key, value in bbox.items():
                    region_metrics[normalized][key].append(value)

        present = set(normalized_labels)
        missing_regions = [label for label in REGION_LABELS if label not in present]
        extra_labels = sorted(label for label in present if label not in REGION_LABELS)

        if not shapes:
            status = "no_shapes"
        elif missing_regions:
            status = "missing_regions"
        elif extra_labels:
            status = "has_extra_labels"
        elif not image_path.exists():
            status = "missing_image_pair"
        else:
            status = "ok"
        status_counts[status] += 1

        file_summaries.append(
            FileSummary(
                json_path=str(json_path),
                image_path=str(image_path),
                image_exists=image_path.exists(),
                image_width=payload.get("imageWidth"),
                image_height=payload.get("imageHeight"),
                shape_count=len(shapes),
                labels="|".join(labels),
                normalized_labels="|".join(normalized_labels),
                has_region_1="区域1" in present,
                has_region_2="区域2" in present,
                has_region_3="区域3" in present,
                missing_regions="|".join(missing_regions),
                extra_labels="|".join(extra_labels),
                status=status,
            )
        )

    orphan_png = sorted(str(path) for key, path in png_keys.items() if key not in json_keys)
    orphan_json = sorted(str(path) for key, path in json_keys.items() if key not in png_keys)

    region_metric_summary = {
        region: {
            metric: summarize_distribution(values)
            for metric, values in metrics.items()
            if not any(math.isnan(v) for v in values)
        }
        for region, metrics in region_metrics.items()
    }

    report = {
        "data_root": str(data_root),
        "total_png": len(png_files),
        "total_json": len(json_files),
        "paired_by_same_folder_and_stem": len(set(png_keys) & set(json_keys)),
        "orphan_png_count": len(orphan_png),
        "orphan_json_count": len(orphan_json),
        "status_counts": dict(status_counts),
        "raw_label_counts": dict(raw_label_counts),
        "normalized_label_counts": dict(normalized_label_counts),
        "shape_type_counts": dict(shape_type_counts),
        "region_metric_summary": region_metric_summary,
        "invalid_json": invalid_json[:100],
        "orphan_png_samples": orphan_png[:100],
        "orphan_json_samples": orphan_json[:100],
    }

    report_path = output_dir / "dataset_stats.json"
    csv_path = output_dir / "file_summary.csv"
    markdown_path = output_dir / "dataset_stats.md"

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(csv_path, file_summaries)

    status_lines = "\n".join(
        f"- {key}: {value}" for key, value in sorted(status_counts.items())
    )
    label_lines = "\n".join(
        f"- {key}: {value}" for key, value in sorted(normalized_label_counts.items())
    )
    metric_lines = "\n".join(
        f"- {region}: width={metrics.get('width')}, height={metrics.get('height')}, area={metrics.get('area')}"
        for region, metrics in sorted(region_metric_summary.items())
    )
    markdown = f"""# Dataset Stats

## Overview

- Data root: `{data_root}`
- PNG files: {len(png_files)}
- JSON files: {len(json_files)}
- Paired files by same folder and stem: {len(set(png_keys) & set(json_keys))}
- Orphan PNG files: {len(orphan_png)}
- Orphan JSON files: {len(orphan_json)}

## File Status

{status_lines}

## Normalized Label Counts

{label_lines}

## Region Shape Metrics

{metric_lines}

## Outputs

- JSON report: `{report_path}`
- Per-file CSV: `{csv_path}`
"""
    markdown_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
