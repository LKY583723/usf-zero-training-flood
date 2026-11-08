"""4 测试场景 × 更紧 post_vv 阈值扫描（conn=500），看紧阈值是否单独救 Jubba_2。

口径：testing_case_256 独立 256² patch，像素聚合 open F1。
"""
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
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
SETTINGS = [-12, -13, -14, -15, -16, -17]


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


acc = {sc: {s: [0, 0, 0] for s in SETTINGS} for sc in SCENES}
for sc in SCENES:
    files = sorted(glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')))
    gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)
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
        base = s['int'] & s['chg']
        post = np.nan_to_num(sar[7])
        for thr in SETTINGS:
            m = conn_filter(base & (post < thr), 500)
            a = acc[sc][thr]
            a[0] += (m & (g == 1)).sum()
            a[1] += (m & (g != 1)).sum()
            a[2] += (~m & (g == 1)).sum()

print('阈值, Weihui, NovaKakhovka, Jubba_1, Jubba_2, Jubba(合并), 3事件均值')
for thr in SETTINGS:
    row = [f'{thr}']
    for sc in SCENES:
        row.append(f'{f1(*acc[sc][thr]):.3f}')
    j = [0, 0, 0]
    for i in range(3):
        j[i] = acc['20231201_Jubba_1'][thr][i] + acc['20231201_Jubba_2'][thr][i]
    row.append(f'{f1(*j):.3f}')
    mean3 = np.mean([f1(*acc['20210727_Weihui'][thr]),
                     f1(*acc['20230609_NovaKakhovka'][thr]), f1(*j)])
    row.append(f'{mean3:.3f}')
    print(', '.join(row))
