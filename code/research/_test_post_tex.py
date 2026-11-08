"""测 post_vv/texture 约束跨事件（分散洪水 NovaKakhovka 重点）。"""
import glob
import os
import re
import sys
import numpy as np
import rasterio
from flood_strategies import all_strategies, conn_filter


def make(sar, post_thr=None, tex_thr=None, conn_min=1500):
    s = all_strategies(sar)
    base = s['int'] & s['chg']
    if post_thr is not None:
        base = base & (np.nan_to_num(sar[7]) < post_thr)
    if tex_thr is not None:
        tex = np.abs(sar[7] - np.nanmean(sar[7]))
        base = base & (tex > tex_thr)
    return conn_filter(base, conn_min)


def test(event):
    PATCHES = f'D:/DisasterM3/UrbanSARFloods/testing_case_256/{event}'
    GT_PATH = f'D:/DisasterM3/UrbanSARFloods/{event}/testing_case_orig/{event}/{event}_GT.tif'
    if not os.path.exists(GT_PATH):
        return

    def parse_id(fn):
        m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
        return int(m.group(1)), int(m.group(2))

    with rasterio.open(GT_PATH) as ds:
        gt_full = ds.read(1)
    files = sorted(glob.glob(os.path.join(PATCHES, '*_SAR.tif')))
    variants = {
        'base+conn': lambda sar: make(sar),
        '+post<-13': lambda sar: make(sar, post_thr=-13),
        '+tex>2.5': lambda sar: make(sar, tex_thr=2.5),
        '+both': lambda sar: make(sar, post_thr=-13, tex_thr=2.5),
    }
    acc = {k: [0, 0, 0] for k in variants}
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
        for name, fn in variants.items():
            m = fn(sar)
            gb = g > 0
            acc[name][0] += (m & gb).sum()
            acc[name][1] += (m & ~gb).sum()
            acc[name][2] += (~m & gb).sum()
    print(f'=== {event} ===')
    for name, (tp, fp, fn) in acc.items():
        p_ = tp / (tp + fp) if tp + fp else 0
        r_ = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
        print(f'  {name}: F1={f1:.3f} P={p_:.3f} R={r_:.3f}')


test('20230609_NovaKakhovka')
test('20210727_Weihui')
test('20231201_Jubba_1')
