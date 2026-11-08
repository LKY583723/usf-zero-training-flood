"""连通域 min_size 扫描：4 事件 × min_size → open F1（patch 级，对齐统一评测）。"""
import glob
import os
import re
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import rasterio
from flood_strategies import all_strategies, conn_filter

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256'
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
MIN_SIZES = [500, 800, 1200, 1500, 2000]


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


print('事件, min_size, openF1, urbanF1, allF1')
for ev in EVENTS:
    patch_dir = os.path.join(PATCHES, ev)
    files = sorted(glob.glob(os.path.join(patch_dir, '*_SAR.tif')))
    if not files:
        continue
    gt_path = os.path.join(DATA, ev, 'testing_case_orig', ev, f'{ev}_GT.tif')
    if not os.path.exists(gt_path):
        continue
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)

    # 每 patch 的 int∩chg 基础掩膜（提前算，避免重复）
    patches = []
    for f in files:
        r, c = parse_id(f)
        y0, x0 = r * 256, c * 256
        if y0 + 256 > gt_full.shape[0] or x0 + 256 > gt_full.shape[1]:
            continue
        g = gt_full[y0:y0 + 256, x0:x0 + 256]
        if (g > 0).sum() < 100:
            continue
        with rasterio.open(f) as ds:
            sar = ds.read().astype(np.float32)
        s = all_strategies(sar)
        patches.append((s['int'] & s['chg'], g))

    for ms in MIN_SIZES:
        acc = {'open': [0,0,0], 'urban': [0,0,0], 'all': [0,0,0]}
        for base, g in patches:
            m = conn_filter(base, ms)
            for cls, gbin in (('open', g==1), ('urban', g==2), ('all', g>0)):
                acc[cls][0] += (m & gbin).sum()
                acc[cls][1] += (m & ~gbin).sum()
                acc[cls][2] += (~m & gbin).sum()
        row = [ev, str(ms)]
        for cls in ['open', 'urban', 'all']:
            tp, fp, fn = acc[cls]
            p_ = tp/(tp+fp) if tp+fp else 0
            r_ = tp/(tp+fn) if tp+fn else 0
            f1 = 2*p_*r_/(p_+r_) if p_+r_ else 0
            row.append(f'{f1:.3f}')
        print(', '.join(row))
