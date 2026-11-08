"""策略层极化组合对比：int/chg 用 VV∩VH（and）vs VV∪VH（or）+ post_vv<-12 + conn。

校准集（train+valid）open F1。目标是看极化交集是否提升精度且不过度伤召回。
"""
import os
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import rasterio
from data_loader import parse_split
from flood_strategies import (strategy_int, strategy_chg, combine_pol,
                              conn_filter, INT_POST_VV)

ROOT = 'D:/DisasterM3/UrbanSARFloods/urban_sar_floods'


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


VARIANTS = ['or_or', 'and_and', 'and_or', 'or_and']
acc = {v: {'open': [0, 0, 0], 'all': [0, 0, 0]} for v in VARIANTS}
n_quad = 0
samples = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')
for i, (sar_p, gt_p) in enumerate(samples):
    with rasterio.open(gt_p) as ds:
        g = ds.read(1)
    H, W = g.shape
    for y0 in range(0, H - 255, 256):
        for x0 in range(0, W - 255, 256):
            q = g[y0:y0 + 256, x0:x0 + 256]
            if (q > 0).sum() < 100:
                continue
            with rasterio.open(sar_p) as ds:
                sar = ds.read(window=((y0, y0 + 256), (x0, x0 + 256))).astype(np.float32)
            for v in VARIANTS:
                im, ic = v.split('_')
                s_int = combine_pol(sar, lambda s, p: strategy_int(s, p), im)
                s_chg = combine_pol(sar, lambda s, p: strategy_chg(s, p), ic)
                base = s_int & s_chg
                post = np.nan_to_num(sar[INT_POST_VV])
                m = conn_filter(base & (post < -12), 500)
                a = acc[v]
                a['open'][0] += (m & (q == 1)).sum(); a['open'][1] += (m & (q != 1)).sum(); a['open'][2] += (~m & (q == 1)).sum()
                a['all'][0] += (m & (q > 0)).sum(); a['all'][1] += (m & (q == 0)).sum(); a['all'][2] += (~m & (q > 0)).sum()
            n_quad += 1
    if (i + 1) % 300 == 0:
        print(f'  [calib] {i+1}/{len(samples)} 样本, {n_quad} 象限', flush=True)

print('int_pol, chg_pol, open F1, all F1, precision, recall')
for v in VARIANTS:
    a = acc[v]
    tp, fp, fn = a['open']
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    im, ic = v.split('_')
    print(f'{im}, {ic}, {f1(tp, fp, fn):.3f}, {f1(*a["all"]):.3f}, {p_:.3f}, {r_:.3f}')
