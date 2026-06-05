"""
投影分析与边界检测。

沿夹线长轴方向做像素强度投影，在投影曲线上检测密度突变点，
结合几何先验确定区域边界。
"""

import cv2
import numpy as np


def _gaussian_smooth(arr_1d: np.ndarray, sigma: float) -> np.ndarray:
    """一维高斯平滑（使用 OpenCV 高斯核）"""
    ksize = int(sigma * 6)
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        ksize = 3
    kernel = cv2.getGaussianKernel(ksize, sigma).flatten()
    return np.convolve(arr_1d, kernel, mode="same")


def compute_projection(arr: np.ndarray, axis: str) -> np.ndarray:
    """
    计算沿指定轴的像素强度投影（均值）。

    Parameters
    ----------
    arr : 2D numpy array
    axis : "vertical" (沿垂直方向投影 → 水平剖面)
           或 "horizontal" (沿水平方向投影 → 垂直剖面)

    Returns
    -------
    1D numpy array, 投影曲线
    """
    if axis == "vertical":
        return arr.mean(axis=0)
    else:
        return arr.mean(axis=1)


def find_boundaries(
    proj: np.ndarray,
    smooth_sigma: float = 15.0,
    min_distance_ratio: float = 0.15,
    n_boundaries: int = 2,
    border_exclude_ratio: float = 0.05,
) -> list[int]:
    """
    在投影曲线上检测密度突变点作为区域边界。

    思路：
    1. 平滑投影曲线，排除图像边界区域
    2. 找到投影曲线的三个主要"高原"（高密度组件区）
    3. 边界放在高原之间的"谷底"

    Parameters
    ----------
    proj : 1D numpy array, 投影曲线
    smooth_sigma : 高斯平滑 sigma（像素）
    min_distance_ratio : 边界之间的最小距离（占总长度的比例）
    n_boundaries : 需要的边界数量（默认 2 → 3 个区域）
    border_exclude_ratio : 排除的图像边界比例（避免边缘强梯度干扰）

    Returns
    -------
    边界位置列表（按升序排列），长度为 n_boundaries
    """
    n = len(proj)
    border_margin = int(n * border_exclude_ratio)
    min_distance = int(n * min_distance_ratio)

    # 平滑
    smoothed = _gaussian_smooth(proj, smooth_sigma)

    # ---- 方法：基于峰的检测 ----
    # 找到投影中所有显著"峰"（高密度区域），在峰之间的最低点放置边界

    # 排除边界区域
    interior_start = border_margin
    interior_end = n - border_margin

    # 计算整体均值作为参考
    global_mean = smoothed[interior_start:interior_end].mean()
    global_range = smoothed[interior_start:interior_end].max() - smoothed[interior_start:interior_end].min()

    # 找投影中的所有局部极大值（峰）
    peaks = []
    for i in range(interior_start + 1, interior_end - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if smoothed[i] > global_mean * 0.7:
                peaks.append((i, smoothed[i]))

    # 找所有局部极小值（谷），并过滤掉浅谷
    valleys = []
    min_valley_depth = global_range * 0.15  # 谷深度阈值：至少是全局范围 15%
    for i in range(interior_start + 1, interior_end - 1):
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            # 检查谷的深度：与相邻最高的峰之差
            left_peak = max([v for p, v in peaks if p < i], default=smoothed[i])
            right_peak = max([v for p, v in peaks if p > i], default=smoothed[i])
            depth = min(left_peak - smoothed[i], right_peak - smoothed[i])
            if depth > min_valley_depth:
                valleys.append((i, smoothed[i]))

    # 如果没有足够的峰和谷，回退到基于梯度的方案
    if len(peaks) < 2 or len(valleys) < n_boundaries:
        return _find_boundaries_by_gradient(
            smoothed, n, border_margin, min_distance, n_boundaries
        )

    # 按位置排序
    peaks.sort()
    valleys.sort()

    # 取最显著的前几个峰（按强度降序），用于定位主组件
    top_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:max(n_boundaries + 1, 3)]
    top_peaks.sort()  # 按位置排序

    # 在这些峰之间找谷，作为边界
    boundaries = []
    for i in range(len(top_peaks) - 1):
        p1_pos = top_peaks[i][0]
        p2_pos = top_peaks[i + 1][0]

        if p2_pos - p1_pos < min_distance:
            continue

        # 在 p1 和 p2 之间找最低的谷
        valley_in_range = [(v_pos, v_val) for v_pos, v_val in valleys
                           if p1_pos < v_pos < p2_pos]

        if valley_in_range:
            best_valley = min(valley_in_range, key=lambda x: x[1])
            boundaries.append(best_valley[0])
        else:
            # 没有谷就用中点
            boundaries.append((p1_pos + p2_pos) // 2)

        if len(boundaries) == n_boundaries:
            break

    # 如果还不足 n_boundaries，补充
    if len(boundaries) < n_boundaries:
        boundaries = _fill_remaining_boundaries(boundaries, n, n_boundaries, min_distance)

    boundaries.sort()
    return boundaries


def _find_boundaries_by_gradient(
    smoothed: np.ndarray,
    n: int,
    border_margin: int,
    min_distance: int,
    n_boundaries: int,
) -> list[int]:
    """基于梯度的回退方案：在内部区域找梯度峰值。"""
    grad = np.gradient(smoothed)
    abs_grad = np.abs(grad)

    # 只考虑内部区域
    interior_grad = abs_grad[border_margin:n - border_margin]
    if interior_grad.std() == 0:
        return _fallback_geometric(n, n_boundaries + 1)[:n_boundaries]

    threshold = float(interior_grad.std() * 2.0)

    candidates = []
    for i in range(border_margin + 1, n - border_margin - 1):
        if abs_grad[i] > threshold and abs_grad[i] >= abs_grad[i - 1] and abs_grad[i] >= abs_grad[i + 1]:
            candidates.append((i, abs_grad[i]))

    candidates.sort(key=lambda x: x[1], reverse=True)

    boundaries = []
    for pos, _ in candidates:
        if all(abs(pos - b) >= min_distance for b in boundaries):
            boundaries.append(pos)
        if len(boundaries) == n_boundaries:
            break

    boundaries.sort()
    if len(boundaries) < n_boundaries:
        boundaries = _fallback_geometric(n, n_boundaries + 1)[:n_boundaries]
    return boundaries


def _fill_remaining_boundaries(
    existing: list[int],
    n: int,
    n_boundaries: int,
    min_distance: int,
) -> list[int]:
    """用几何分割补充不足的边界。"""
    fallback = _fallback_geometric(n, n_boundaries + 1)
    for b in fallback:
        if len(existing) >= n_boundaries:
            break
        if all(abs(b - e) >= min_distance for e in existing):
            existing.append(b)
    return sorted(existing)


def _fallback_geometric(n_pixels: int, n_regions: int) -> list[int]:
    """几何回退：均匀分三段。"""
    return [int(n_pixels * (i + 1) / (n_regions + 1)) for i in range(n_regions)]


def compute_texture_score(arr_roi: np.ndarray) -> float:
    """
    计算区域的纹理得分（检测股线周期性条纹）。

    对 ROI 的每一行做 FFT，计算高频能量占比。
    铝绞线区有明显股线纹理 → 高频能量高 → score 高。

    Parameters
    ----------
    arr_roi : 2D numpy array, 区域子图

    Returns
    -------
    float, 纹理得分（0~1，越高纹理越明显）
    """
    h, w = arr_roi.shape
    if h < 10 or w < 10:
        return 0.0

    # 取中间 50% 行，避免边缘干扰
    start_row = h // 4
    end_row = 3 * h // 4
    if end_row <= start_row:
        start_row, end_row = 0, h

    scores = []
    for row in range(start_row, end_row, max(1, (end_row - start_row) // 10)):
        line = arr_roi[row, :].astype(np.float64)
        line = line - line.mean()
        if line.std() == 0:
            continue
        fft = np.abs(np.fft.rfft(line))
        # 纹理频率范围：排除 DC 和极低频（前 2% 频段），也排除极高频（后 20%）
        n_freqs = len(fft)
        low_cut = max(1, int(n_freqs * 0.02))
        high_cut = int(n_freqs * 0.8)
        if high_cut <= low_cut:
            continue
        texture_energy = fft[low_cut:high_cut].sum()
        total_energy = fft[low_cut:].sum()
        if total_energy > 0:
            scores.append(texture_energy / total_energy)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def refine_boundary_with_texture(
    arr: np.ndarray,
    boundaries: list[int],
    axis: str,
) -> list[int]:
    """
    利用纹理分析精调区域2/3边界。

    思路：铝绞线区（区域3）有明显的股线周期性纹理（FFT 高频能量高），
    在边界候选附近搜索纹理从低到高的突变点。

    Parameters
    ----------
    arr : 2D numpy array
    boundaries : 当前边界列表 [b1, b2]
    axis : "vertical" 或 "horizontal"

    Returns
    -------
    精调后的边界列表
    """
    if len(boundaries) != 2:
        return boundaries

    b1, b2 = boundaries
    n = arr.shape[0] if axis == "vertical" else arr.shape[1]

    # 搜索范围：b2 上下各 10%（相对于区域2宽度），但至少 15px，最多 50px
    search_half = max(15, min(50, (b2 - b1) // 10))
    window_h = max(15, (b2 - b1) // 5)

    # 搜集搜索范围内每个位置的纹理得分
    candidates = []
    search_start = max(b1 + window_h, b2 - search_half)
    search_end = min(n - window_h, b2 + search_half)

    for candidate in range(search_start, search_end, 3):
        roi_up = arr[candidate - window_h:candidate, :]
        roi_down = arr[candidate:candidate + window_h, :]

        tex_up = compute_texture_score(roi_up)
        tex_down = compute_texture_score(roi_down)

        candidates.append((candidate, tex_up, tex_down, tex_down - tex_up))

    if not candidates:
        return boundaries

    # 只在与原边界接近的位置找纹理突变
    # 要求 tex_down > tex_up（下方纹理更强 = 铝绞线区）
    tex_increases = [(c, tu, td, td - tu) for c, tu, td, d in candidates if td > tu]

    if tex_increases:
        # 选纹理差异最大的位置
        best = max(tex_increases, key=lambda x: x[3])
        new_b2 = best[0]
    else:
        # 如果找不到纹理增加的点，保持原边界
        new_b2 = b2

    # 不过度偏离原始边界
    if abs(new_b2 - b2) > search_half:
        new_b2 = b2

    return [b1, new_b2]
