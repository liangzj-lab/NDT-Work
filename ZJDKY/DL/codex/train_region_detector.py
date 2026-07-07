"""Train a Faster R-CNN detector for tension clamp regions 1/2/3."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image  # noqa: F401  # Preload Pillow DLLs before torch/torchvision on Windows.
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights

from prepare_region_data import main as prepare_data_main
from region_dataset import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    ID_TO_LABEL,
    NUM_CLASSES,
    RegionDetectionDataset,
    collate_fn,
    load_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-prepare", action="store_true")
    return parser.parse_args()


def build_model(min_size: int, max_size: int, pretrained_backbone: bool = False):
    backbone_weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    try:
        return fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=NUM_CLASSES,
            min_size=min_size,
            max_size=max_size,
        )
    except TypeError:
        return fasterrcnn_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=pretrained_backbone,
            num_classes=NUM_CLASSES,
            min_size=min_size,
            max_size=max_size,
        )


def box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    if box_a.numel() == 0 or box_b.numel() == 0:
        return torch.zeros((box_a.shape[0], box_b.shape[0]), device=box_a.device)
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a[:, None] + area_b - inter
    return inter / union.clamp(min=1e-6)


def ap_from_detections(detections: list[tuple[float, int]]) -> float:
    if not detections:
        return 0.0
    detections = sorted(detections, key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    precision_recall: list[tuple[float, float]] = []
    total_gt = sum(is_tp for _, is_tp in detections)
    if total_gt == 0:
        return 0.0
    for _, is_tp in detections:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / total_gt
        precision_recall.append((precision, recall))

    ap = 0.0
    previous_recall = 0.0
    for precision, recall in precision_recall:
        ap += precision * max(0.0, recall - previous_recall)
        previous_recall = recall
    return ap


@torch.no_grad()
def evaluate(model, loader, device: torch.device, score_threshold: float, iou_threshold: float) -> dict:
    model.eval()
    class_detections = {label_id: [] for label_id in ID_TO_LABEL}
    matched_ious: list[float] = []
    per_image_status: list[dict] = []

    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].detach().cpu()
            pred_labels = output["labels"].detach().cpu()
            pred_scores = output["scores"].detach().cpu()
            gt_boxes = target["boxes"].detach().cpu()
            gt_labels = target["labels"].detach().cpu()

            image_status = {"missing": [], "duplicates": []}
            for label_id in ID_TO_LABEL:
                gt_for_label = gt_boxes[gt_labels == label_id]
                keep = (pred_labels == label_id) & (pred_scores >= score_threshold)
                pred_for_label = pred_boxes[keep]
                score_for_label = pred_scores[keep]
                if len(pred_for_label) == 0:
                    image_status["missing"].append(ID_TO_LABEL[label_id])
                    class_detections[label_id].append((0.0, 0))
                    continue
                if len(pred_for_label) > 1:
                    image_status["duplicates"].append(ID_TO_LABEL[label_id])

                best_idx = int(torch.argmax(score_for_label).item())
                best_box = pred_for_label[best_idx:best_idx + 1]
                score = float(score_for_label[best_idx].item())
                iou = float(box_iou(best_box, gt_for_label).max().item())
                matched_ious.append(iou)
                class_detections[label_id].append((score, int(iou >= iou_threshold)))
            if image_status["missing"] or image_status["duplicates"]:
                per_image_status.append(image_status)

    per_class_ap50 = {
        ID_TO_LABEL[label_id]: ap_from_detections(class_detections[label_id])
        for label_id in ID_TO_LABEL
    }
    map50 = sum(per_class_ap50.values()) / max(len(per_class_ap50), 1)
    mean_iou = sum(matched_ious) / max(len(matched_ious), 1)
    return {
        "map50": round(map50, 6),
        "mean_iou": round(mean_iou, 6),
        "per_class_ap50": per_class_ap50,
        "problem_image_count": len(per_image_status),
    }


def prepare_if_needed(args: argparse.Namespace) -> None:
    if args.no_prepare and (args.output_dir / "splits.json").exists():
        return
    import sys

    original_argv = sys.argv[:]
    sys.argv = [
        "prepare_region_data.py",
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
    ]
    if args.limit is not None:
        sys.argv += ["--limit", str(args.limit)]
    try:
        prepare_data_main()
    finally:
        sys.argv = original_argv


def write_training_log(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.device == "cpu" or not args.device.startswith("cuda"):
        raise RuntimeError("Training must use GPU. Please set --device cuda or cuda:<index>.")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA, but torch.cuda.is_available() is False.")

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_if_needed(args)

    splits = load_splits(args.output_dir / "splits.json")
    train_dataset = RegionDetectionDataset(splits["train"], hflip_prob=args.hflip_prob)
    val_dataset = RegionDetectionDataset(splits["val"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device(args.device)
    model = build_model(args.min_size, args.max_size, args.pretrained_backbone).to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    best_map50 = -1.0
    best_mean_iou = -1.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        total_loss = 0.0
        step_count = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += float(losses.item())
            step_count += 1
        scheduler.step()

        metrics = evaluate(
            model,
            val_loader,
            device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
        )
        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(step_count, 1), 6),
            "map50": metrics["map50"],
            "mean_iou": metrics["mean_iou"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 2),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "label_map": {str(k): v for k, v in ID_TO_LABEL.items()},
            "model_args": {"min_size": args.min_size, "max_size": args.max_size},
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if (
            metrics["map50"] > best_map50
            or (metrics["map50"] == best_map50 and metrics["mean_iou"] > best_mean_iou)
        ):
            best_map50 = metrics["map50"]
            best_mean_iou = metrics["mean_iou"]
            torch.save(checkpoint, args.output_dir / "best.pt")

    final_metrics = {
        "best_map50": best_map50,
        "best_mean_iou": best_mean_iou,
        "last": history[-1] if history else {},
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_training_log(args.output_dir / "training_log.csv", history)
    print(f"Wrote {args.output_dir / 'best.pt'}")
    print(f"Wrote {args.output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
