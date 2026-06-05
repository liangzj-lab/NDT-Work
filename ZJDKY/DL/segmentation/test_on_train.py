"""在 Data/Train 标注数据上测试分割算法。"""
import json, cv2, numpy as np, os, sys
from pathlib import Path

base = Path('d:/MyCode/NDT-work/ZJDKY')
sys.path.insert(0, str(base))

from DL.segmentation.region_splitter import RegionSplitter, _to_8bit
from DL.segmentation.visualize import save_visualization

train_dir = base / 'Data' / 'Train'
out_dir = base / 'Data' / 'Train_segmented'
out_dir.mkdir(parents=True, exist_ok=True)

# 找到所有 PNG/JSON 对
pairs = []
for f in os.listdir(str(train_dir)):
    if f.endswith('.png'):
        png_path = train_dir / f
        json_path = train_dir / (f.replace('.png', '.json'))
        if json_path.exists():
            pairs.append((png_path, json_path))

print(f'Found {len(pairs)} image/label pairs\n')

splitter = RegionSplitter(smooth_sigma=20.0)
results = []

def imread_unicode(filepath: Path):
    """OpenCV imread 不支持中文路径，用 numpy 绕开。"""
    data = np.fromfile(str(filepath), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img

for png_path, json_path in sorted(pairs, key=lambda x: x[0].name):
    img = imread_unicode(png_path)
    if img is None:
        print(f'[SKIP] Cannot read: {png_path.name}')
        continue
    arr = img.astype(np.float64) * 256

    # GT annotation
    with open(str(json_path), 'r', encoding='utf-8') as fp:
        gt = json.load(fp)
    gt_clamp = None
    gt_regions = {}
    for s in gt['shapes']:
        pts = np.array(s['points'])
        if s['label'] == '线夹':
            gt_clamp = (float(pts[:, 1].min()), float(pts[:, 1].max()))
        elif s['label'] in ('区域1', '区域2', '区域3'):
            rname = 'region_' + s['label'][-1]
            gt_regions[rname] = (float(pts[:, 0].min()), float(pts[:, 0].max()))

    # Prediction
    try:
        result = splitter.split(arr)
    except Exception as e:
        print(f'[ERROR] {png_path.name}: {e}')
        continue

    y1, y2 = result['_clamp_y_range']
    bounds = result['_boundaries_x']
    h, w = arr.shape

    # Evaluate clamp y detection
    clamp_y_err = None
    if gt_clamp:
        clamp_y_err = abs(y1 - gt_clamp[0])
        clamp_y_err += abs(y2 - gt_clamp[1])
        clamp_y_err /= 2

    # Evaluate region boundaries
    # GT regions: x ranges. Compare with predicted boundaries.
    gt_x_bounds = []
    for rname in ['region_1', 'region_2', 'region_3']:
        if rname in gt_regions:
            gt_x_bounds.append(gt_regions[rname][0])
            gt_x_bounds.append(gt_regions[rname][1])
    gt_x_bounds = sorted(set(gt_x_bounds))

    print(f'{png_path.name}:')
    print(f'  GT clamp y: {gt_clamp}, Pred: ({y1},{y2}), err={clamp_y_err:.0f}px' if clamp_y_err else f'  Pred clamp: ({y1},{y2})')
    print(f'  GT region x ranges: {gt_regions}')
    print(f'  Pred boundaries x: {bounds}')

    # Compute boundary error: closest pred boundary to each GT boundary
    if len(bounds) >= 2 and len(gt_x_bounds) >= 4:
        # GT region boundaries are internal boundaries (between regions), not outer edges
        # Extract interior boundaries
        gt_inner = sorted(set([gt_regions[r][1] for r in gt_regions] + [gt_regions[r][0] for r in gt_regions]))
        # Remove outer edges
        all_x = []
        for r in gt_regions.values():
            all_x.extend(r)
        gt_inner = sorted(set(all_x))
        if len(gt_inner) >= 4:
            gt_inner = gt_inner[1:-1]  # remove outermost
        gt_inner = sorted(set(gt_inner))

        # Match each pred boundary to closest GT inner boundary
        for b in bounds:
            closest = min(gt_inner, key=lambda x: abs(x - b))
            print(f'    pred b={b}: closest GT={closest:.0f}, err={abs(b-closest):.0f}px')

    results.append({'file': png_path.name, 'clamp_y_err': clamp_y_err, 'bounds': bounds, 'gt_regions': gt_regions})

    # Save output (cv2.imwrite doesn't support Chinese paths either)
    def imwrite_unicode(filepath: Path, img):
        ext = filepath.suffix
        ret, buf = cv2.imencode(ext, img)
        if ret:
            buf.tofile(str(filepath))

    stem = png_path.stem
    vis_dir = out_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis = result['_visual_8bit']
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    # draw on vis_rgb
    y1_d, y2_d = result['_clamp_y_range']
    cv2.rectangle(vis_rgb, (0, y1_d), (vis_rgb.shape[1]-1, y2_d), (0, 255, 255), 2)
    for bx in result['_boundaries_x']:
        cv2.line(vis_rgb, (bx, y1_d), (bx, y2_d), (0, 255, 255), 3)
        cv2.line(vis_rgb, (bx, y1_d), (bx, y2_d), (0, 0, 0), 1)
    # Labels
    labels = ['R1:钢芯锚固', 'R2:铝压接', 'R3:铝绞线']
    colors = [(0,255,0), (255,0,0), (0,0,255)]
    bounds = result['_boundaries_x']
    w_d = vis_rgb.shape[1]
    centers = [bounds[0]//2, (bounds[0]+bounds[1])//2, (bounds[1]+w_d)//2]
    for cx, lb, col in zip(centers, labels, colors):
        (tw, th), _ = cv2.getTextSize(lb, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        tx = max(5, min(cx-tw//2, w_d-tw-5))
        ty = y1_d + 30
        cv2.rectangle(vis_rgb, (tx-5, ty-th-5), (tx+tw+5, ty+5), (0,0,0), -1)
        cv2.putText(vis_rgb, lb, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2)
    imwrite_unicode(vis_dir / f'{stem}_seg.png', vis_rgb)

    crops_dir = out_dir / 'crops'
    crops_dir.mkdir(parents=True, exist_ok=True)
    for rname in ['region_1', 'region_2', 'region_3']:
        crop = result[rname]
        crop_8bit = _to_8bit(crop)
        imwrite_unicode(crops_dir / f'{stem}_{rname}.png', crop_8bit)

print(f'\nDone. {len(results)}/{len(pairs)} processed.')
