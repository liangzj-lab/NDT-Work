"""
批量区域分割处理脚本。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pydicom

from .region_splitter import RegionSplitter, _to_8bit
from .visualize import save_visualization


def process_dcm(
    dcm_path: Path,
    output_dir: Path,
    splitter: RegionSplitter,
) -> dict | None:
    """处理单张 DICOM，返回结果信息。"""
    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array.astype(np.float64)
    except Exception as e:
        print(f"  [ERROR] Read failed: {e}")
        return None

    try:
        result = splitter.split(arr)
    except Exception as e:
        print(f"  [ERROR] Split failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    y1, y2 = result["_clamp_y_range"]
    bounds = result["_boundaries_x"]
    vis = result["_visual_8bit"]
    stem = dcm_path.stem

    # 保存 crop
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for rname in ["region_1", "region_2", "region_3"]:
        crop = result[rname]
        crop_8bit = _to_8bit(crop)
        cv2.imwrite(str(crops_dir / f"{stem}_{rname}.png"), crop_8bit)

    # 保存可视化
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_visualization(vis, (y1, y2), bounds, vis_dir / f"{stem}_seg.png")

    return {
        "file": stem,
        "clamp_y": (y1, y2),
        "boundaries_x": bounds,
    }


def process_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    pattern: str = "*.dcm",
    smooth_sigma: float = 12.0,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Not found: {input_dir}")

    if output_dir is None:
        output_dir = input_dir.parent / (input_dir.name + "_segmented")

    files = sorted(input_dir.glob(pattern))
    if not files:
        print(f"No files matching '{pattern}' in {input_dir}")
        return

    print(f"Found {len(files)} file(s)")
    print(f"Output: {output_dir}\n")

    splitter = RegionSplitter(smooth_sigma=smooth_sigma)
    results = []
    for f in files:
        print(f"[{f.name}]")
        r = process_dcm(f, output_dir, splitter)
        if r:
            results.append(r)
            print(f"  clamp y=({r['clamp_y'][0]},{r['clamp_y'][1]}), bounds x={r['boundaries_x']}")

    print(f"\nDone. {len(results)}/{len(files)} succeeded.")

    # 如有标注，评估准确率
    eval_segmentation(input_dir, results, output_dir)


def eval_segmentation(
    input_dir: Path,
    results: list[dict],
    output_dir: Path,
) -> None:
    """对比 JSON 标注评估分割准确率。"""
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        return

    print("\n--- Evaluation vs Ground Truth ---")
    # 构建标注查找表
    gt_map = {}
    for jf in json_files:
        with open(str(jf), "r", encoding="utf-8") as fp:
            data = json.load(fp)
        stem = jf.stem
        shapes = data.get("shapes", [])
        gt_map[stem] = {"clamp": None, "regions": {}}
        for s in shapes:
            pts = np.array(s["points"])
            xs, ys = pts[:, 0], pts[:, 1]
            if s["label"] == "线夹":
                gt_map[stem]["clamp"] = (float(ys.min()), float(ys.max()))
            elif s["label"] in ("区域1", "区域2", "区域3"):
                rname = "region_" + s["label"][-1]
                gt_map[stem]["regions"][rname] = (float(xs.min()), float(xs.max()))

    for r in results:
        stem = r["file"]
        if stem not in gt_map:
            continue
        gt = gt_map[stem]
        clamp_err = abs(r["clamp_y"][0] - gt["clamp"][0]) if gt["clamp"] else None
        print(f"  {stem}: clamp_y_err={clamp_err:.0f}px" if clamp_err else f"  {stem}")

        # 对比区域边界（只比较 x 坐标，因为 clamp y 已对齐）
        if len(r["boundaries_x"]) == 2 and len(gt["regions"]) >= 2:
            for rname, (gt_x1, gt_x2) in gt["regions"].items():
                # 找到对应预测区域（按内容标注后顺序可能不同，这里只比较边界位置）
                pass
            # 简化：比较是否有2条边界
            print(f"    GT regions: {len(gt['regions'])}, Pred boundaries: {r['boundaries_x']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="耐张线夹区域分割批量处理")
    parser.add_argument("--input_dir", type=Path, default=Path("Data/Train"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--smooth_sigma", type=float, default=20.0)
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, smooth_sigma=args.smooth_sigma)


if __name__ == "__main__":
    main()
