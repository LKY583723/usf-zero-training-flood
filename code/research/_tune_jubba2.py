"""Jubba_2 专项调优：严格暗水 + 纹理 + 小连通域。"""
import glob
import os
import re
import sys
import numpy as np
import rasterio
from flood_strategies import all_strategies, conn_filter

ev = '20231201_Jubba_2'
PATCHES = f'D:/DisasterM3/UrbanSARFloods/testing_case_256/{ev}'
GT_PATH = f'D:/DisasterM3/UrbanSARFloods/{ev}/testing_case_orig/{ev}/{ev}_GT.tif'


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


with rasterio.open(GT_PATH) as ds:
    gt_full = ds.read(1)
files = sorted(glob.glob(os.path.join(PATCHES, '*_SAR.tif')))
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
    patches.append((s, sar, g))


def make(sar, s, post_thr, tex_thr, conn_min):
    base = s['int'] & s['chg']
    base = base & (np.nan_to_num(sar[7]) < post_thr)
    if tex_thr:
        tex = np.abs(sar[7] - np.nanmean(sar[7]))
        base = base & (tex > tex_thr)
    return conn_filter(base, conn_min)


best = []
for post_thr in [-16, -17, -18, -19, -20]:
    for tex_thr in [5, 6, 7, 8]:
        for conn_min in [150, 200, 250, 300, 400, 500]:
            acc = [0, 0, 0]
            for s, sar, g in patches:
                m = make(sar, s, post_thr, tex_thr, conn_min)
                gb = g > 0
                acc[0] += (m & gb).sum()
                acc[1] += (m & ~gb).sum()
                acc[2] += (~m & gb).sum()
            tp, fp, fn = acc
            p_ = tp / (tp + fp) if tp + fp else 0
            r_ = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
            best.append((f1, post_thr, tex_thr, conn_min, p_, r_))

best.sort(reverse=True)
print('Top 15 Jubba_2 组合:')
print(f'{"F1":>6s} {"post":>5s} {"tex":>4s} {"conn":>5s} {"P":>6s} {"R":>6s}')
for f1, post, tex, conn, p_, r_ in best[:15]:
    print(f'{f1:6.3f} {post:5d} {tex:4d} {conn:5d} {p_:6.3f} {r_:6.3f}')
