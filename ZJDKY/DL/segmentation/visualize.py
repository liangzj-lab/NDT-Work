"""
分割结果可视化。
在夹线 ROI 区域叠加边界线和标签。
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def draw_result(
    img_8bit: np.ndarray,
    clamp_y_range: tuple[int, int],
    boundaries_x: list[int],
    labels: list[str] | None = None,
) -> np.ndarray:
    """
    绘制分割结果。

    Parameters
    ----------
    img_8bit : (H, W) 灰度图
    clamp_y_range : (y1, y2) 夹线 ROI
    boundaries_x : [x1, x2] 两条垂直边界
    labels : 三区域标签
    """
    if labels is None:
        labels = ["R1: 钢芯锚固", "R2: 铝压接", "R3: 铝绞线"]

    vis = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
    h, w = vis.shape[:2]
    y1, y2 = clamp_y_range
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

    # 画夹线 ROI 框
    cv2.rectangle(vis, (0, y1), (w - 1, y2), (0, 255, 255), 2)

    # 画垂直边界线
    for bx in boundaries_x:
        cv2.line(vis, (bx, y1), (bx, y2), (0, 255, 255), 3)
        cv2.line(vis, (bx, y1), (bx, y2), (0, 0, 0), 1)

    # 区域标签
    region_centers_x = [
        boundaries_x[0] // 2,
        (boundaries_x[0] + boundaries_x[1]) // 2,
        (boundaries_x[1] + w) // 2,
    ]
    for i, (cx, label) in enumerate(zip(region_centers_x, labels)):
        ty = y1 + 30
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        tx = max(5, min(cx - tw // 2, w - tw - 5))
        cv2.rectangle(vis, (tx - 5, ty - th - 5), (tx + tw + 5, ty + 5), (0, 0, 0), -1)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.0, colors[i], 2)

    return vis


def save_visualization(
    img_8bit: np.ndarray,
    clamp_y_range: tuple[int, int],
    boundaries_x: list[int],
    output_path: Path | str,
) -> None:
    """生成可视化图并保存。"""
    vis = draw_result(img_8bit, clamp_y_range, boundaries_x)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
