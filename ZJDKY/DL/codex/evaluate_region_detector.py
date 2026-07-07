"""Evaluate and visualize a trained region detector on a saved split."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch

from predict_regions import load_model
from region_dataset import DEFAULT_OUTPUT_DIR, ID_TO_LABEL, RegionSample, load_splits, pil_to_float_tensor
from train_region_detector import box_iou


PRED_COLORS = {
    1: (46, 204, 113),
    2: (52, 152, 219),
    3: (231, 76, 60),
}
GT_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_OUTPUT_DIR / "best.pt")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_OUTPUT_DIR / "splits.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "test_predictions")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--candidate-score-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--selection", default="score", choices=("score", "geometry"))
    parser.add_argument("--max-candidates-per-label", type=int, default=8)
    parser.add_argument("--merge-aligned-candidates", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    return parser.parse_args()


def load_font(size: int = 22):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: tuple[int, int, int], width: int) -> None:
    draw.rectangle([float(value) for value in box], outline=color, width=width)


def draw_visualization(sample: RegionSample, rows: list[dict], out_path: Path) -> None:
    image = Image.open(sample.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font()

    for label_id, box in zip(sample.labels, sample.boxes):
        draw_box(draw, box, GT_COLOR, 2)
        draw.text((box[0] + 6, box[1] + 6), f"GT{label_id}", fill=GT_COLOR, font=font)

    for row in rows:
        label_id = int(row["label_id"])
        color = PRED_COLORS.get(label_id, (255, 255, 0))
        box = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
        draw_box(draw, box, color, 5)
        draw.text(
            (box[0] + 6, box[1] + 32),
            f"P{label_id} {float(row['score']):.2f} IoU {float(row['iou']):.2f}",
            fill=color,
            font=font,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def y_overlap_ratio(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    top = max(float(box_a[1]), float(box_b[1]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    overlap = max(0.0, bottom - top)
    min_height = max(1.0, min(float(box_a[3] - box_a[1]), float(box_b[3] - box_b[1])))
    return overlap / min_height


def geometry_combo_score(combo: tuple[tuple[torch.Tensor, float, int], ...]) -> float:
    boxes = [item[0] for item in combo]
    scores = [item[1] for item in combo]
    centers_x = [float((box[0] + box[2]) / 2) for box in boxes]
    centers_y = [float((box[1] + box[3]) / 2) for box in boxes]
    heights = [max(1.0, float(box[3] - box[1])) for box in boxes]

    score_term = sum(math.log(max(score, 1e-6)) for score in scores)
    region2_between = min(centers_x[0], centers_x[2]) <= centers_x[1] <= max(centers_x[0], centers_x[2])
    between_term = 3.0 if region2_between else -8.0
    y_spread = (max(centers_y) - min(centers_y)) / max(sum(heights) / len(heights), 1.0)
    y_overlap = (
        y_overlap_ratio(boxes[0], boxes[1])
        + y_overlap_ratio(boxes[1], boxes[2])
        + y_overlap_ratio(boxes[0], boxes[2])
    ) / 3.0
    return score_term + between_term - (2.5 * y_spread) + (2.0 * y_overlap)


def candidates_by_label(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    score_threshold: float,
    max_candidates_per_label: int,
) -> dict[int, list[tuple[torch.Tensor, float, int]]]:
    grouped: dict[int, list[tuple[torch.Tensor, float, int]]] = {}
    for label_id in sorted(ID_TO_LABEL):
        label_candidates: list[tuple[torch.Tensor, float, int]] = []
        keep_indexes = torch.nonzero((pred_labels == label_id) & (pred_scores >= score_threshold)).flatten()
        for index in keep_indexes.tolist():
            label_candidates.append((pred_boxes[index], float(pred_scores[index].item()), int(index)))
        label_candidates.sort(key=lambda item: item[1], reverse=True)
        grouped[label_id] = label_candidates[:max_candidates_per_label]
    return grouped


def select_candidates(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    score_threshold: float,
    selection: str,
    max_candidates_per_label: int,
) -> dict[int, tuple[torch.Tensor, float, int]]:
    grouped = candidates_by_label(
        pred_boxes,
        pred_labels,
        pred_scores,
        score_threshold,
        max_candidates_per_label,
    )
    if selection == "score" or any(not grouped[label_id] for label_id in ID_TO_LABEL):
        return {
            label_id: candidates[0]
            for label_id, candidates in grouped.items()
            if candidates
        }

    labels = sorted(ID_TO_LABEL)
    best_combo = None
    best_score = float("-inf")
    for combo in itertools.product(*(grouped[label_id] for label_id in labels)):
        combo_score = geometry_combo_score(combo)
        if combo_score > best_score:
            best_score = combo_score
            best_combo = combo
    return {
        label_id: candidate
        for label_id, candidate in zip(labels, best_combo or [])
    }


def merge_aligned_candidates(
    selected: dict[int, tuple[torch.Tensor, float, int]],
    grouped: dict[int, list[tuple[torch.Tensor, float, int]]],
    min_y_overlap: float = 0.50,
) -> dict[int, tuple[torch.Tensor, float, int]]:
    merged = {}
    for label_id, (selected_box, selected_score, selected_index) in selected.items():
        boxes_to_merge = [selected_box]
        for candidate_box, candidate_score, candidate_index in grouped.get(label_id, []):
            if candidate_index == selected_index:
                continue
            if y_overlap_ratio(selected_box, candidate_box) >= min_y_overlap:
                boxes_to_merge.append(candidate_box)
        if len(boxes_to_merge) == 1:
            merged[label_id] = (selected_box, selected_score, selected_index)
            continue
        stacked = torch.stack(boxes_to_merge)
        union_box = torch.tensor(
            [
                float(stacked[:, 0].min()),
                float(stacked[:, 1].min()),
                float(stacked[:, 2].max()),
                float(stacked[:, 3].max()),
            ],
            dtype=selected_box.dtype,
        )
        merged[label_id] = (union_box, selected_score, selected_index)
    return merged


@torch.no_grad()
def evaluate_sample(
    model,
    sample: RegionSample,
    device: torch.device,
    score_threshold: float,
    candidate_score_threshold: float,
    iou_threshold: float,
    selection: str,
    max_candidates_per_label: int,
    merge_aligned: bool,
) -> tuple[list[dict], list[dict]]:
    image = Image.open(sample.image_path).convert("RGB")
    tensor = pil_to_float_tensor(image).to(device)
    output = model([tensor])[0]

    pred_boxes = output["boxes"].detach().cpu()
    pred_labels = output["labels"].detach().cpu()
    pred_scores = output["scores"].detach().cpu()
    gt_boxes = torch.as_tensor(sample.boxes, dtype=torch.float32)
    gt_labels = torch.as_tensor(sample.labels, dtype=torch.int64)

    rows: list[dict] = []
    anomalies: list[dict] = []
    grouped = candidates_by_label(
        pred_boxes,
        pred_labels,
        pred_scores,
        candidate_score_threshold,
        max_candidates_per_label,
    )
    selected = select_candidates(
        pred_boxes,
        pred_labels,
        pred_scores,
        candidate_score_threshold,
        selection,
        max_candidates_per_label,
    )
    if merge_aligned:
        selected = merge_aligned_candidates(selected, grouped)
    for label_id in sorted(ID_TO_LABEL):
        keep = (pred_labels == label_id) & (pred_scores >= score_threshold)
        boxes_for_label = pred_boxes[keep]
        candidate_count = int(
            len(torch.nonzero((pred_labels == label_id) & (pred_scores >= candidate_score_threshold)).flatten())
        )
        gt_for_label = gt_boxes[gt_labels == label_id]

        if label_id not in selected:
            anomalies.append(
                {
                    "image_path": sample.image_path,
                    "label_id": label_id,
                    "label_name": ID_TO_LABEL[label_id],
                    "issue": "missing_region",
                    "score": "",
                    "iou": 0.0,
                }
            )
            continue

        best_box, score, _ = selected[label_id]
        iou = float(box_iou(best_box.unsqueeze(0), gt_for_label).max().item())
        x1, y1, x2, y2 = [float(value) for value in best_box.tolist()]
        row = {
            "image_path": sample.image_path,
            "label_id": label_id,
            "label_name": ID_TO_LABEL[label_id],
            "score": round(score, 6),
            "iou": round(iou, 6),
            "pass_iou": int(iou >= iou_threshold),
            "prediction_count_for_label": int(len(boxes_for_label)),
            "candidate_count_for_label": candidate_count,
            "x_min": round(x1, 2),
            "y_min": round(y1, 2),
            "x_max": round(x2, 2),
            "y_max": round(y2, 2),
        }
        rows.append(row)

        if iou < iou_threshold:
            anomalies.append(
                {
                    "image_path": sample.image_path,
                    "label_id": label_id,
                    "label_name": ID_TO_LABEL[label_id],
                    "issue": "low_iou",
                    "score": round(score, 6),
                    "iou": round(iou, 6),
                }
            )

    return rows, anomalies


@torch.no_grad()
def collect_candidates(
    model,
    sample: RegionSample,
    device: torch.device,
    score_threshold: float,
    max_candidates_per_label: int,
) -> list[dict]:
    image = Image.open(sample.image_path).convert("RGB")
    tensor = pil_to_float_tensor(image).to(device)
    output = model([tensor])[0]
    pred_boxes = output["boxes"].detach().cpu()
    pred_labels = output["labels"].detach().cpu()
    pred_scores = output["scores"].detach().cpu()
    gt_boxes = torch.as_tensor(sample.boxes, dtype=torch.float32)
    gt_labels = torch.as_tensor(sample.labels, dtype=torch.int64)

    rows: list[dict] = []
    grouped = candidates_by_label(
        pred_boxes,
        pred_labels,
        pred_scores,
        score_threshold,
        max_candidates_per_label,
    )
    for label_id, candidates in grouped.items():
        gt_for_label = gt_boxes[gt_labels == label_id]
        for rank, (box, score, model_rank) in enumerate(candidates):
            iou = float(box_iou(box.unsqueeze(0), gt_for_label).max().item())
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            rows.append(
                {
                    "image_path": sample.image_path,
                    "label_id": label_id,
                    "label_name": ID_TO_LABEL[label_id],
                    "candidate_rank_for_label": rank,
                    "model_rank": model_rank,
                    "score": round(score, 6),
                    "iou": round(iou, 6),
                    "x_min": round(x1, 2),
                    "y_min": round(y1, 2),
                    "x_max": round(x2, 2),
                    "y_max": round(y2, 2),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], anomalies: list[dict], sample_count: int, split_name: str) -> dict:
    by_label = {}
    for label_id in sorted(ID_TO_LABEL):
        label_rows = [row for row in rows if int(row["label_id"]) == label_id]
        ious = [float(row["iou"]) for row in label_rows]
        by_label[ID_TO_LABEL[label_id]] = {
            "detected": len(label_rows),
            "expected": sample_count,
            "pass_iou50": sum(int(row["pass_iou"]) for row in label_rows),
            "mean_iou": round(sum(ious) / max(len(ious), 1), 6),
            "min_iou": round(min(ious), 6) if ious else 0.0,
            "mean_score": round(
                sum(float(row["score"]) for row in label_rows) / max(len(label_rows), 1),
                6,
            ),
        }

    passed_regions = sum(int(row["pass_iou"]) for row in rows)
    expected_regions = sample_count * len(ID_TO_LABEL)
    return {
        "split": split_name,
        "image_count": sample_count,
        "expected_region_count": expected_regions,
        "detected_region_count": len(rows),
        "pass_iou50_region_count": passed_regions,
        "region_iou50_accuracy": round(passed_regions / max(expected_regions, 1), 6),
        "mean_iou": round(sum(float(row["iou"]) for row in rows) / max(len(rows), 1), 6),
        "anomaly_count": len(anomalies),
        "per_class": by_label,
    }


def main() -> int:
    args = parse_args()
    if args.device == "cpu" or not args.device.startswith("cuda"):
        raise RuntimeError("Evaluation must use GPU. Please set --device cuda or cuda:<index>.")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA, but torch.cuda.is_available() is False.")

    splits = load_splits(args.split_file)
    samples = splits[args.split]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_model(args.weights, device, args.min_size, args.max_size)
    candidate_score_threshold = (
        args.candidate_score_threshold
        if args.candidate_score_threshold is not None
        else args.score_threshold
    )

    all_rows: list[dict] = []
    all_anomalies: list[dict] = []
    all_candidates: list[dict] = []
    viz_dir = args.output_dir / "visualizations"
    for index, sample in enumerate(samples):
        rows, anomalies = evaluate_sample(
            model,
            sample,
            device,
            score_threshold=args.score_threshold,
            candidate_score_threshold=candidate_score_threshold,
            iou_threshold=args.iou_threshold,
            selection=args.selection,
            max_candidates_per_label=args.max_candidates_per_label,
            merge_aligned=args.merge_aligned_candidates,
        )
        candidates = collect_candidates(
            model,
            sample,
            device,
            score_threshold=candidate_score_threshold,
            max_candidates_per_label=args.max_candidates_per_label,
        )
        all_rows.extend(rows)
        all_anomalies.extend(anomalies)
        all_candidates.extend(candidates)
        draw_visualization(sample, rows, viz_dir / f"{index:04d}_{Path(sample.image_path).stem}.png")

    summary = summarize(all_rows, all_anomalies, len(samples), args.split)
    summary.update(
        {
            "weights": str(args.weights),
            "split_file": str(args.split_file),
            "score_threshold": args.score_threshold,
            "candidate_score_threshold": candidate_score_threshold,
            "iou_threshold": args.iou_threshold,
            "selection": args.selection,
            "merge_aligned_candidates": args.merge_aligned_candidates,
        }
    )

    write_csv(
        args.output_dir / "predictions.csv",
        all_rows,
        [
            "image_path",
            "label_id",
            "label_name",
            "score",
            "iou",
            "pass_iou",
            "prediction_count_for_label",
            "candidate_count_for_label",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ],
    )
    write_csv(
        args.output_dir / "prediction_anomalies.csv",
        all_anomalies,
        ["image_path", "label_id", "label_name", "issue", "score", "iou"],
    )
    write_csv(
        args.output_dir / "candidates.csv",
        all_candidates,
        [
            "image_path",
            "label_id",
            "label_name",
            "candidate_rank_for_label",
            "model_rank",
            "score",
            "iou",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_dir / 'predictions.csv'}")
    print(f"Wrote {args.output_dir / 'prediction_anomalies.csv'}")
    print(f"Wrote {args.output_dir / 'candidates.csv'}")
    print(f"Wrote {viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
