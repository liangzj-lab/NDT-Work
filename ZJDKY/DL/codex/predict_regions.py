"""Predict tension clamp region boxes with a trained detector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from region_postprocess import select_candidates
from region_dataset import DEFAULT_OUTPUT_DIR, ID_TO_LABEL, NUM_CLASSES, pil_to_float_tensor
from train_region_detector import build_model


COLORS = {
    1: (46, 204, 113),
    2: (52, 152, 219),
    3: (231, 76, 60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--candidate-score-threshold", type=float, default=0.05)
    parser.add_argument("--selection", default="geometry", choices=("score", "geometry"))
    parser.add_argument("--max-candidates-per-label", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    return parser.parse_args()


def discover_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    )


def load_model(weights_path: Path, device: torch.device, min_size: int | None, max_size: int | None):
    checkpoint = torch.load(weights_path, map_location=device)
    model_args = checkpoint.get("model_args", {})
    model_min_size = min_size or int(model_args.get("min_size", 512))
    model_max_size = max_size or int(model_args.get("max_size", 1024))
    model = build_model(model_min_size, model_max_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def draw_predictions(image_path: Path, rows: list[dict], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    for row in rows:
        label_id = int(row["label_id"])
        color = COLORS.get(label_id, (255, 255, 0))
        box = [
            float(row["x_min"]),
            float(row["y_min"]),
            float(row["x_max"]),
            float(row["y_max"]),
        ]
        draw.rectangle(box, outline=color, width=5)
        text = f"{row['label_name']} {float(row['score']):.2f}"
        draw.text((box[0] + 6, box[1] + 6), text, fill=color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


@torch.no_grad()
def predict_image(
    model,
    image_path: Path,
    device: torch.device,
    score_threshold: float,
    candidate_score_threshold: float,
    selection: str,
    max_candidates_per_label: int,
) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    tensor = pil_to_float_tensor(image).to(device)
    output = model([tensor])[0]
    pred_boxes = output["boxes"].detach().cpu()
    pred_labels = output["labels"].detach().cpu()
    pred_scores = output["scores"].detach().cpu()
    selected = select_candidates(
        pred_boxes,
        pred_labels,
        pred_scores,
        sorted(ID_TO_LABEL),
        candidate_score_threshold,
        selection,
        max_candidates_per_label,
    )
    rows: list[dict] = []
    for label_id in sorted(ID_TO_LABEL):
        if label_id not in selected:
            continue
        box, score_value, _ = selected[label_id]
        if score_value < score_threshold and selection == "score":
            continue
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        rows.append(
            {
                "image_path": str(image_path),
                "label_id": label_id,
                "label_name": ID_TO_LABEL[label_id],
                "score": round(score_value, 6),
                "x_min": round(x1, 2),
                "y_min": round(y1, 2),
                "x_max": round(x2, 2),
                "y_max": round(y2, 2),
            }
        )
    return rows


def anomaly_for_rows(image_path: Path, rows: list[dict]) -> list[dict]:
    anomalies: list[dict] = []
    by_label: dict[int, int] = {label_id: 0 for label_id in ID_TO_LABEL}
    for row in rows:
        by_label[int(row["label_id"])] += 1
    for label_id, count in by_label.items():
        if count == 0:
            anomalies.append(
                {
                    "image_path": str(image_path),
                    "label_name": ID_TO_LABEL[label_id],
                    "issue": "missing_region",
                    "count": count,
                }
            )
        elif count > 1:
            anomalies.append(
                {
                    "image_path": str(image_path),
                    "label_name": ID_TO_LABEL[label_id],
                    "issue": "duplicate_region",
                    "count": count,
                }
            )
    return anomalies


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.weights, device, args.min_size, args.max_size)
    image_paths = discover_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found under {args.input}")

    all_rows: list[dict] = []
    all_anomalies: list[dict] = []
    viz_dir = args.output_dir / "visualizations" / "predictions"
    for index, image_path in enumerate(image_paths):
        rows = predict_image(
            model,
            image_path,
            device,
            args.score_threshold,
            args.candidate_score_threshold,
            args.selection,
            args.max_candidates_per_label,
        )
        all_rows.extend(rows)
        all_anomalies.extend(anomaly_for_rows(image_path, rows))
        draw_predictions(image_path, rows, viz_dir / f"pred_{index:04d}_{image_path.stem}.png")

    predictions_path = args.output_dir / "predictions.csv"
    anomalies_path = args.output_dir / "prediction_anomalies.csv"
    summary_path = args.output_dir / "prediction_summary.json"
    write_csv(
        predictions_path,
        all_rows,
        ["image_path", "label_id", "label_name", "score", "x_min", "y_min", "x_max", "y_max"],
    )
    write_csv(
        anomalies_path,
        all_anomalies,
        ["image_path", "label_name", "issue", "count"],
    )
    summary = {
        "weights": str(args.weights),
        "input": str(args.input),
        "image_count": len(image_paths),
        "prediction_count": len(all_rows),
        "anomaly_count": len(all_anomalies),
        "score_threshold": args.score_threshold,
        "candidate_score_threshold": args.candidate_score_threshold,
        "selection": args.selection,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {predictions_path}")
    print(f"Wrote {anomalies_path}")
    print(f"Wrote {viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
