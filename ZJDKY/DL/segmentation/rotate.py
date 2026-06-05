"""
方向检测与旋转校正。

通过 Canny 边缘 + Hough 直线检测确定图像中夹线的长轴方向，
必要时将图像旋转至水平/垂直方向以便后续投影分析。
"""

import cv2
import numpy as np


def detect_orientation(arr_8bit: np.ndarray) -> dict:
    """
    检测图像中结构的主方向。

    Parameters
    ----------
    arr_8bit : uint8 numpy array (H, W)

    Returns
    -------
    dict with keys:
      - angle_deg : 主方向角度（度），0°=水平
      - confidence : 置信度 (0~1)
      - need_rotation : 是否需要旋转校正
    """
    edges = cv2.Canny(arr_8bit, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

    if lines is None:
        return {"angle_deg": 0.0, "confidence": 0.0, "need_rotation": False}

    angles = []
    for rho, theta in lines[:, 0, :]:
        angle = np.rad2deg(theta)
        # 归一化到 [-90, 90)
        if angle >= 90:
            angle -= 180
        angles.append(angle)

    angles = np.array(angles)

    # 使用中位数作为主方向
    median_angle = float(np.median(angles))

    # 置信度：角度集中程度
    mad = float(np.median(np.abs(angles - median_angle)))
    confidence = max(0.0, 1.0 - mad / 45.0)

    # 如果主方向接近水平（0°）或垂直（±90°），无需旋转
    need_rotation = abs(median_angle) > 5.0 and abs(median_angle) < 85.0

    return {
        "angle_deg": median_angle,
        "confidence": confidence,
        "need_rotation": need_rotation,
    }


def rotate_to_horizontal(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    将图像旋转，使其主方向变为水平。

    Parameters
    ----------
    arr : numpy array (H, W)
    angle_deg : 当前主方向角度

    Returns
    -------
    旋转后的数组
    """
    h, w = arr.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 计算旋转后的新边界
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]

    rotated = cv2.warpAffine(arr, matrix, (new_w, new_h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    return rotated


def find_long_axis(arr: np.ndarray) -> str:
    """
    通过投影分析确定长轴方向（结构变化最大的方向）。

    Parameters
    ----------
    arr : numpy array (H, W), 可以是 16-bit 或 8-bit

    Returns
    -------
    "horizontal" 或 "vertical"
    """
    proj_h = arr.mean(axis=0)  # 沿水平方向的投影
    proj_v = arr.mean(axis=1)  # 沿垂直方向的投影

    # 哪个方向的变化更大，哪个方向就是长轴
    if proj_h.std() > proj_v.std():
        return "horizontal"
    else:
        return "vertical"
