"""
耐张线夹区域分割主模块。

流程:
  1. 行投影 → 定位夹线水平条带 ROI（排除上下干扰）
  2. 列投影 → 在 ROI 内找两谷底 → 三区域分界
  3. 按内容标注 → 区域1(钢芯，最亮最窄)、区域2(铝压接)、区域3(铝绞线，最宽纹理最强)

对外接口：RegionSplitter 类 和 split_regions 便捷函数
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pydicom

from .projection import _gaussian_smooth, compute_texture_score


class RegionSplitter:
    """耐张线夹区域分割器。"""

    def __init__(
        self,
        smooth_sigma: float = 12.0,
        min_region_ratio: float = 0.10,
    ):
        """
        Parameters
        ----------
        smooth_sigma : 投影曲线平滑 sigma
        min_region_ratio : 区域最小宽度（占夹线总宽比例）
        """
        self.smooth_sigma = smooth_sigma
        self.min_region_ratio = min_region_ratio

        # 结果
        self.clamp_y_range_: Optional[Tuple[int, int]] = None
        self.boundaries_x_: Optional[list[int]] = None  # 在完整图像中的 x 坐标
        self.regions_: Optional[Dict[str, np.ndarray]] = None

    def split(self, pixel_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Parameters
        ----------
        pixel_array : 2D numpy array (H, W), 原始 16-bit 像素

        Returns
        -------
        dict: region_1, region_2, region_3, _visual_8bit, _clamp_roi, _boundaries_x
        """
        arr = pixel_array.astype(np.float64)
        h, w = arr.shape

        # ---- Step 1: 定位夹线水平条带 ----
        y1, y2 = self._find_clamp_band(arr)
        self.clamp_y_range_ = (y1, y2)
        clamp_strip = arr[y1:y2, :]
        # 稍微扩展以包含完整夹线
        margin = (y2 - y1) // 4
        y1 = max(0, y1 - margin)
        y2 = min(h, y2 + margin)
        clamp_strip = arr[y1:y2, :]

        # ---- Step 2: 列投影 → 找三区域边界 ----
        col_proj = clamp_strip.mean(axis=0)
        valleys = self._find_valleys_in_projection(col_proj, n_boundaries=2)

        # 如果只找到 1 个谷 → 用纹理分析找第二个边界
        if len(valleys) < 2:
            tex_boundary = self._find_texture_boundary(clamp_strip, valleys[0] if valleys else None)
            if tex_boundary is not None:
                valleys.append(tex_boundary)
                valleys.sort()

        boundaries = valleys
        self.boundaries_x_ = boundaries

        # ---- Step 3: Crop 三段 ----
        segments = [
            arr[y1:y2, 0:boundaries[0]],
            arr[y1:y2, boundaries[0]:boundaries[1]],
            arr[y1:y2, boundaries[1]:w],
        ]

        # ---- Step 4: 按内容标注 ----
        labeled = _label_by_content(segments)

        # 可视化
        arr_8bit = _to_8bit(arr)
        labeled["_visual_8bit"] = arr_8bit
        labeled["_clamp_y_range"] = (y1, y2)
        labeled["_boundaries_x"] = boundaries

        self.regions_ = labeled
        return labeled

    # ------------------------------------------------------------------
    # 夹线 ROI 检测
    # ------------------------------------------------------------------

    def _find_clamp_band(self, arr: np.ndarray) -> Tuple[int, int]:
        """
        行投影找夹线水平条带。

        夹线特征：图像下半部的横向长亮带，内部有密度起伏。
        策略：在图像下部 60% 区域找最强亮带，用双阈值扩展。
        """
        h, w = arr.shape
        row_mean = arr.mean(axis=1)
        smoothed = _gaussian_smooth(row_mean, sigma=10.0)
        row_range = smoothed.max() - smoothed.min()

        # 只在图像下半部（40%-100%）搜索夹线，避开上方干扰
        search_start = int(h * 0.40)
        search_end = h

        # 在搜索区域内找峰值
        peak_row = search_start + int(np.argmax(smoothed[search_start:search_end]))
        peak_val = smoothed[peak_row]

        # 扩展：向上/下找到投影降至峰值 30% 的位置
        drop_threshold = peak_val * 0.30
        y1 = max(0, peak_row - 250)
        for i in range(peak_row, search_start, -1):
            if smoothed[i] < drop_threshold:
                y1 = i
                break

        y2 = min(h, peak_row + 250)
        for i in range(peak_row, search_end):
            if smoothed[i] < drop_threshold:
                y2 = i
                break

        # 约束高度
        if y2 - y1 < 80:
            y1 = max(0, peak_row - 100)
            y2 = min(h, peak_row + 100)
        if y2 - y1 > 700:
            y2 = y1 + 500

        return y1, y2

    # ------------------------------------------------------------------
    # 区域边界检测
    # ------------------------------------------------------------------

    def _find_valleys_in_projection(
        self, proj: np.ndarray, n_boundaries: int = 2
    ) -> list[int]:
        """
        在水平投影曲线上找谷底作为区域边界。

        方法：
        1. 平滑投影曲线
        2. 找显著的谷（局部极小值，且足够深）
        3. 取最深的 n_boundaries 个谷
        """
        n = len(proj)
        smoothed = _gaussian_smooth(proj, sigma=self.smooth_sigma)
        proj_range = smoothed.max() - smoothed.min()
        if proj_range == 0:
            return [int(n * (i + 1) / (n_boundaries + 1)) for i in range(n_boundaries)]

        min_distance = int(n * self.min_region_ratio)

        # 找所有局部极小值（谷），深度阈值 2%
        valleys = []
        for i in range(min_distance, n - min_distance):
            if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                left_max = smoothed[max(0, i - min_distance):i].max()
                right_max = smoothed[i + 1:min(n, i + min_distance + 1)].max()
                depth = min(left_max - smoothed[i], right_max - smoothed[i])
                if depth > proj_range * 0.02:
                    valleys.append((i, depth))

        valleys.sort(key=lambda x: x[1], reverse=True)

        boundaries = []
        for pos, _ in valleys:
            if all(abs(pos - b) >= min_distance for b in boundaries):
                boundaries.append(pos)
            if len(boundaries) == n_boundaries:
                break

        boundaries.sort()

        if len(boundaries) < n_boundaries:
            boundaries = [int(n * (i + 1) / (n_boundaries + 1)) for i in range(n_boundaries)]

        return boundaries

    def _find_texture_boundary(
        self, clamp_strip: np.ndarray, known_boundary: int | None
    ) -> int | None:
        """
        利用纹理分析找第二个区域边界（铝压接↔铝绞线）。

        铝绞线区有明显的股线纹理（FFT 高频能量高），
        铝压接区纹理较少。沿水平方向滑动窗口，找纹理突变点。
        """
        h_strip, w_strip = clamp_strip.shape
        if w_strip < 300 or h_strip < 20:
            return None

        window_w = max(50, w_strip // 20)
        step = window_w // 2

        # 计算每个窗口的纹理得分
        scores = []
        for x_start in range(0, w_strip - window_w, step):
            roi = clamp_strip[:, x_start:x_start + window_w]
            tex = compute_texture_score(roi)
            scores.append((x_start + window_w // 2, tex))

        if len(scores) < 3:
            return None

        positions = np.array([s[0] for s in scores])
        tex_vals = np.array([s[1] for s in scores])

        # 平滑纹理曲线
        smoothed_tex = _gaussian_smooth(tex_vals, sigma=2.0)
        tex_range = smoothed_tex.max() - smoothed_tex.min()
        if tex_range == 0:
            return None

        # 找纹理从低到高的最大跳变点
        tex_grad = np.gradient(smoothed_tex)
        # 找最大正梯度位置（纹理从低变高）
        best_idx = int(np.argmax(tex_grad))
        best_pos = int(positions[best_idx])

        # 如果已知一个边界，确保纹理边界不在它太近的位置
        if known_boundary is not None:
            min_dist = w_strip * self.min_region_ratio
            if abs(best_pos - known_boundary) < min_dist:
                # 尝试另一个方向
                if best_idx < len(positions) - 1:
                    best_pos = int(positions[-1])
                else:
                    best_pos = int(positions[0])

        return best_pos


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _to_8bit(arr: np.ndarray) -> np.ndarray:
    """16-bit → 8-bit 映射。"""
    vmin = np.percentile(arr, 0.5)
    vmax = np.percentile(arr, 99.5)
    if vmax == vmin:
        vmax = vmin + 1
    img = (arr - vmin) / (vmax - vmin) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _label_by_content(segments: list[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    按内容将三个片段标注为 region_1/2/3。

    规则（基于7张标注数据的统计规律）：
    - region_1 (钢芯锚固): 最亮 + 最窄
    - region_3 (铝绞线): 最宽 (纹理最强)
    - region_2 (铝压接): 剩下的
    """
    if len(segments) != 3:
        return {f"region_{i+1}": seg for i, seg in enumerate(segments)}

    # 计算特征
    feats = []
    for i, seg in enumerate(segments):
        mask = seg > seg.max() * 0.05
        valid = seg[mask]
        mean_val = float(valid.mean()) if len(valid) > 0 else float(seg.mean())
        tex = compute_texture_score(seg)
        width = seg.shape[1]
        feats.append({
            "idx": i,
            "mean": mean_val,
            "texture": tex,
            "width": width,
            "crop": seg,
        })

    # 区域1 = 最亮的（钢芯）
    by_bright = sorted(feats, key=lambda x: x["mean"], reverse=True)
    r1_idx = by_bright[0]["idx"]

    # 区域3 = 最宽的（铝绞线，纹理也最强）
    # 在剩下的两个中选宽度最大的
    remaining = [f for f in feats if f["idx"] != r1_idx]
    by_width = sorted(remaining, key=lambda x: x["width"], reverse=True)
    r3_idx = by_width[0]["idx"]

    # 区域2 = 最后一个
    r2_idx = [f["idx"] for f in feats if f["idx"] not in (r1_idx, r3_idx)][0]

    labeled = {}
    for target_name, idx in [("region_1", r1_idx), ("region_2", r2_idx), ("region_3", r3_idx)]:
        labeled[target_name] = feats[idx]["crop"]

    return labeled


# ------------------------------------------------------------------
# 便捷接口
# ------------------------------------------------------------------

def split_regions(
    dcm_path: Path | str,
    smooth_sigma: float = 12.0,
) -> Dict[str, np.ndarray]:
    """读 DICOM → 区域分割。"""
    ds = pydicom.dcmread(str(dcm_path))
    pixel_array = ds.pixel_array

    splitter = RegionSplitter(smooth_sigma=smooth_sigma)
    return splitter.split(pixel_array)
