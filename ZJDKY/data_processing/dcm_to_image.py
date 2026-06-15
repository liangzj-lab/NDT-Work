"""
DICOM (.dcm) 转 JPG/PNG，支持自动调整窗宽窗位 + 图像增强。

窗宽窗位模式:
  - auto: 使用 DICOM 标签中的 WindowCenter/WindowWidth；若无标签，自动计算最优窗宽窗位
  - percentile: 基于百分位数自动计算（默认 2%~98%）
  - minmax: 直接映射 [min, max] → [0, 255]
  - otsu: 大津法（Otsu）自动阈值分割，适合双峰分布图像

图像增强模式:
  - gamma: Gamma 校正，调亮暗区结构，gamma < 1 提亮，> 1 压暗（默认）
  - clahe: 自适应直方图均衡化（CLAHE），提升局部对比度
  - ndt: NDT 优化管线 — 双边滤波去噪 → CLAHE → 轻度锐化
  - sharpen: 反锐化掩模（Unsharp Masking），增强边缘/结构边界
  - none: 不做增强

用法:
  python data_processing/dcm_to_image.py
  python data_processing/dcm_to_image.py --input_dir Data
  python data_processing/dcm_to_image.py --input_dir Data --enhance clahe --gamma 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pydicom
from PIL import Image

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


# ---------------------------------------------------------------------------
# 窗宽窗位计算
# ---------------------------------------------------------------------------

def apply_window(pixel_array: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    将 DICOM 像素值通过窗宽窗位映射到 [0, 255] 的 8-bit 图像。

    Parameters
    ----------
    pixel_array : 原始像素数组 (int16/uint16)
    window_center : 窗位（WL），即窗口中心对应的原始像素值
    window_width : 窗宽（WW），即窗口覆盖的像素值范围

    Returns
    -------
    uint8 numpy array
    """
    # 窗口上下界
    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0

    # 线性映射并 clamp
    img = (pixel_array - lower) / (upper - lower) * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _window_from_tags(ds: pydicom.Dataset) -> Optional[Tuple[float, float]]:
    """从 DICOM 标签读取 WindowCenter / WindowWidth。"""
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is None or ww is None:
        return None
    # 可能为多值（取第一个）
    if isinstance(wc, (list, tuple, pydicom.multival.MultiValue)):
        wc = wc[0]
    if isinstance(ww, (list, tuple, pydicom.multival.MultiValue)):
        ww = ww[0]
    return float(wc), float(ww)


def _window_minmax(pixel_array: np.ndarray) -> tuple[float, float]:
    """窗宽窗位 = [min, max] 全范围。"""
    vmin, vmax = pixel_array.min(), pixel_array.max()
    center = (vmin + vmax) / 2.0
    width = float(vmax - vmin)
    if width == 0:
        width = 1.0
    return center, width


def _window_percentile(pixel_array: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    """
    基于百分位数计算窗宽窗位，自动排除离群像素。

    Parameters
    ----------
    low : 下百分位（默认 2%）
    high : 上百分位（默认 98%）
    """
    vmin = np.percentile(pixel_array, low)
    vmax = np.percentile(pixel_array, high)
    center = (vmin + vmax) / 2.0
    width = float(vmax - vmin)
    if width == 0:
        width = 1.0
    return center, width


def _window_otsu(pixel_array: np.ndarray) -> tuple[float, float]:
    """
    大津法（Otsu）自动计算窗宽窗位。
    找到最佳分割阈值作为窗位，窗宽取前景/背景各 3 倍标准差。
    """
    # 将数据归一化到 [0, 255] 用于直方图
    vmin, vmax = pixel_array.min(), pixel_array.max()
    if vmax == vmin:
        return float(vmin), 1.0

    hist, bin_edges = np.histogram(pixel_array, bins=256, range=(vmin, vmax))
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)

    max_between = 0
    best_thresh = 0
    w_b = 0
    sum_b = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f

        between = w_b * w_f * (m_b - m_f) ** 2
        if between > max_between:
            max_between = between
            best_thresh = t

    # 将阈值映射回原始像素值
    center = vmin + best_thresh / 255.0 * (vmax - vmin)

    # 窗宽取前景和背景各自 2 倍标准差的较大值 × 2
    fg = pixel_array[pixel_array > center]
    bg = pixel_array[pixel_array <= center]
    std_fg = float(fg.std()) if len(fg) > 0 else 0.0
    std_bg = float(bg.std()) if len(bg) > 0 else 0.0
    width = max(std_fg, std_bg) * 4.0
    if width == 0:
        width = float(vmax - vmin)

    return center, width


# ---------------------------------------------------------------------------
# 图像增强
# ---------------------------------------------------------------------------

def _require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required for enhance modes 'clahe', 'sharpen', and 'ndt'. "
            "Install it in the dcm_convert environment with: "
            "conda install -n dcm_convert -c conda-forge opencv"
        )


def enhance_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    CLAHE（Contrast Limited Adaptive Histogram Equalization）
    自适应直方图均衡化 —— 在局部窗口内做直方图均衡，并限制对比度放大倍数，
    避免噪声也被放大。

    Parameters
    ----------
    img : uint8 numpy array (H, W)
    clip_limit : 对比度裁剪阈值（默认 2.0），越大局部对比越强
    tile_size : 局部窗口大小（默认 8），越小越局部

    Returns
    -------
    uint8 numpy array
    """
    _require_cv2()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


def enhance_gamma(img: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """
    Gamma 校正。
    gamma < 1：提亮暗区（让暗部结构更可见）
    gamma > 1：压暗亮区
    gamma = 1：不变

    公式: out = 255 * (in / 255) ^ gamma
    """
    img_float = img.astype(np.float64) / 255.0
    corrected = np.power(img_float, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def enhance_sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    反锐化掩模（Unsharp Masking）锐化。

    Parameters
    ----------
    img : uint8 numpy array
    strength : 锐化强度（默认 1.0），越大边缘越锐利

    流程: 高斯模糊 → 原图 + strength * (原图 - 模糊图)
    """
    _require_cv2()
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    detail = img.astype(np.float32) - blurred.astype(np.float32)
    sharpened = img.astype(np.float32) + strength * detail
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_ndt(img: np.ndarray) -> np.ndarray:
    """
    NDT 射线检测图像专用增强管线，旨在提升焊缝结构细节的可见性：

    1. 双边滤波 — 降噪保边（去除超声散斑噪声，同时保留缺陷边缘）
    2. CLAHE — 局部对比度增强（让暗区/亮区的细微结构同时可见）
    3. 轻度锐化 — 增强结构边界

    适用于射线无损检测图像（铝焊缝等）。
    """
    # 1) 双边滤波：保边去噪，去除超声散斑
    _require_cv2()
    denoised = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)

    # 2) CLAHE：局部对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 3) 反锐化掩模：轻度锐化，突出结构边缘
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    detail = enhanced.astype(np.float32) - blurred.astype(np.float32)
    sharpened = enhanced.astype(np.float32) + 0.6 * detail
    enhanced = np.clip(sharpened, 0, 255).astype(np.uint8)

    return enhanced


def apply_enhance(img: np.ndarray, enhance_mode: str, **kwargs: float) -> np.ndarray:
    """
    应用图像增强。

    Parameters
    ----------
    img : uint8 numpy array (H, W)
    enhance_mode : "none" | "clahe" | "gamma" | "sharpen" | "ndt"
    kwargs : 传递给具体增强函数的参数
    """
    if enhance_mode == "none":
        return img
    elif enhance_mode == "clahe":
        return enhance_clahe(img, **{k: v for k, v in kwargs.items() if k in ("clip_limit", "tile_size")} or {})
    elif enhance_mode == "gamma":
        return enhance_gamma(img, **{k: v for k, v in kwargs.items() if k == "gamma"} or {})
    elif enhance_mode == "sharpen":
        return enhance_sharpen(img, **{k: v for k, v in kwargs.items() if k == "strength"} or {})
    elif enhance_mode == "ndt":
        return enhance_ndt(img)
    else:
        raise ValueError(f"Unknown enhance mode: {enhance_mode}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def convert_dcm_to_image(
    dcm_path: Path,
    output_dir: Path,
    output_format: str = "png",
    mode: str = "auto",
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    enhance: str = "gamma",
    enhance_kwargs: Optional[dict] = None,
) -> Path:
    """
    将单个 DICOM 文件转换为 JPG/PNG。

    Parameters
    ----------
    dcm_path : DICOM 文件路径
    output_dir : 输出目录
    output_format : "jpg" 或 "png"
    mode : 窗宽窗位模式 — "auto", "percentile", "minmax", "otsu"
    percentile_low / percentile_high : percentile 模式下的上下百分位
    enhance : 增强模式 — "none", "clahe", "gamma", "sharpen", "ndt"
    enhance_kwargs : 传递给增强函数的额外参数

    Returns
    -------
    输出文件路径
    """
    ds = pydicom.dcmread(str(dcm_path))
    pixel_array = ds.pixel_array.astype(np.float64)

    # ---- 选择窗宽窗位 ----
    window_center, window_width = None, None

    if mode == "auto":
        result = _window_from_tags(ds)
        if result is not None:
            window_center, window_width = result
        else:
            window_center, window_width = _window_percentile(pixel_array, low=2, high=98)
    elif mode == "percentile":
        window_center, window_width = _window_percentile(pixel_array, percentile_low, percentile_high)
    elif mode == "minmax":
        window_center, window_width = _window_minmax(pixel_array)
    elif mode == "otsu":
        window_center, window_width = _window_otsu(pixel_array)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"  [{dcm_path.name}] window_center={window_center:.1f}, window_width={window_width:.1f}, enhance={enhance}")

    # ---- 应用窗宽窗位 ----
    img_8bit = apply_window(pixel_array, window_center, window_width)

    # ---- 处理 MONOCHROME1（反色）----
    photometric = getattr(ds, "PhotometricInterpretation", "")
    if "MONOCHROME1" in str(photometric):
        img_8bit = 255 - img_8bit

    # ---- 图像增强 ----
    if enhance != "none":
        kwargs = enhance_kwargs or {}
        img_8bit = apply_enhance(img_8bit, enhance, **kwargs)

    # ---- 保存图像 ----
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = dcm_path.stem + f".{output_format}"
    output_path = output_dir / output_name

    img = Image.fromarray(img_8bit, mode="L")
    if output_format == "jpg":
        img.save(str(output_path), quality=95)
    else:
        img.save(str(output_path))

    return output_path


def main() -> None:
    # Edit these values to run the script directly from an IDE.
    input_dir = Path(r"D:\项目文件\执行项目文件\PG25-LX19 浙江锅检所铝焊缝智能超声检测技术研究\执行过程文件\线夹图像\2026.3.28日天津检测220kV东范一线")
    output_dir = None
    output_format = "png"
    window_mode = "auto"
    enhance_mode = "gamma"
    gamma = 0.7
    percentile_low = 2.0
    percentile_high = 98.0
    clahe_clip = 2.0
    clahe_tile = 8
    sharpen_strength = 1.0

    parser = argparse.ArgumentParser(
        description="DICOM (.dcm) 转 JPG/PNG，支持自动窗宽窗位调整 + 图像增强"
    )
    parser.add_argument(
        "--input_dir", type=Path, default=input_dir,
        help="DICOM 文件所在目录（默认: Data）"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=output_dir,
        help="输出目录（默认: 在 input_dir 同级自动生成 {input_dir_name}_png/）"
    )
    parser.add_argument(
        "--format", choices=["jpg", "png"], default=output_format,
        help="输出格式（默认: png）"
    )
    parser.add_argument(
        "--mode", choices=["auto", "percentile", "minmax", "otsu"], default=window_mode,
        help="窗宽窗位模式（默认: auto）"
    )
    parser.add_argument(
        "--percentile_low", type=float, default=percentile_low,
        help="percentile 模式下的下百分位（默认: 2）"
    )
    parser.add_argument(
        "--percentile_high", type=float, default=percentile_high,
        help="percentile 模式下的上百分位（默认: 98）"
    )
    parser.add_argument(
        "--enhance", choices=["none", "clahe", "gamma", "sharpen", "ndt"],
        default=enhance_mode,
        help="图像增强模式（默认: gamma）"
    )
    parser.add_argument(
        "--gamma", type=float, default=gamma,
        help="gamma 校正值，<1 提亮暗区（默认: 0.7），仅 --enhance gamma 时生效"
    )
    parser.add_argument(
        "--clahe_clip", type=float, default=clahe_clip,
        help="CLAHE 对比度裁剪阈值（默认: 2.0），仅 --enhance clahe 时生效"
    )
    parser.add_argument(
        "--clahe_tile", type=int, default=clahe_tile,
        help="CLAHE 窗口大小（默认: 8），仅 --enhance clahe 时生效"
    )
    parser.add_argument(
        "--sharpen_strength", type=float, default=sharpen_strength,
        help="锐化强度（默认: 1.0），仅 --enhance sharpen 时生效"
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    # 自动推导输出目录：在 input_dir 同级生成 {input_dir_name}_png/
    if args.output_dir is None:
        output_dir = args.input_dir.parent / (args.input_dir.name + "_png")
    else:
        output_dir = args.output_dir

    dcm_files = sorted(
        path for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".dcm"
    )
    if not dcm_files:
        print(f"No .dcm files found in {args.input_dir}")
        return

    # 构建增强参数
    enhance_kwargs = {}
    if args.enhance == "gamma":
        enhance_kwargs["gamma"] = args.gamma
    elif args.enhance == "clahe":
        enhance_kwargs["clip_limit"] = args.clahe_clip
        enhance_kwargs["tile_size"] = args.clahe_tile
    elif args.enhance == "sharpen":
        enhance_kwargs["strength"] = args.sharpen_strength

    print(f"Found {len(dcm_files)} DICOM file(s) under {args.input_dir}")
    print(f"Mode: {args.mode}, Enhance: {args.enhance}, Format: {args.format}, Output: {output_dir}\n")

    for dcm_file in dcm_files:
        relative_parent = dcm_file.parent.relative_to(args.input_dir)
        current_output_dir = output_dir / relative_parent
        output_path = convert_dcm_to_image(
            dcm_path=dcm_file,
            output_dir=current_output_dir,
            output_format=args.format,
            mode=args.mode,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            enhance=args.enhance,
            enhance_kwargs=enhance_kwargs,
        )
        print(f"  -> saved: {output_path}")

    print(f"\nDone. Converted {len(dcm_files)} file(s).")


if __name__ == "__main__":
    main()
