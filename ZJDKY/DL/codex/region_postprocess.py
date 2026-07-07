"""Post-processing helpers for fixed 1/2/3 region detector outputs."""

from __future__ import annotations

import itertools
import math

import torch


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
    label_ids: list[int],
    score_threshold: float,
    max_candidates_per_label: int,
) -> dict[int, list[tuple[torch.Tensor, float, int]]]:
    grouped: dict[int, list[tuple[torch.Tensor, float, int]]] = {}
    for label_id in sorted(label_ids):
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
    label_ids: list[int],
    score_threshold: float,
    selection: str,
    max_candidates_per_label: int,
) -> dict[int, tuple[torch.Tensor, float, int]]:
    grouped = candidates_by_label(
        pred_boxes,
        pred_labels,
        pred_scores,
        label_ids,
        score_threshold,
        max_candidates_per_label,
    )
    if selection == "score" or any(not grouped[label_id] for label_id in label_ids):
        return {
            label_id: candidates[0]
            for label_id, candidates in grouped.items()
            if candidates
        }

    labels = sorted(label_ids)
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
