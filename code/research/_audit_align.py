"""审计：patch 与 GT 对齐是否精确（位移扫描）。
如果对齐正确，config F 的 open F1 应在位移 (0,0) 处取最大；
若存在系统性偏移，峰值会出现在非零位移。
同时检查 GT 值分布与来源合理性。
"""
import glob
import os
import re
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import rasterio
from flood_strategies import all_strategies, conn_filter, strategy_int, strategy_chg

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256'
CONN_MIN = 500
POST_THR = -12.0


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def config_f(sar):
    s = all_strategies(sar)
    f = s['int'] & s['chg']
    f = f & (np.nan_to_num(sar[7]) < POST_THR)
    return conn_filter(f, CONN_MIN)


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def check_event(ev, n_patches=8, shift_range=12):
    files = sorted(glob.glob(os.path.join(PATCHES, ev, '*_SAR.tif')))
    gt_path = os.path.join(DATA, ev, 'testing_case_orig', ev, f'{ev}_GT.tif')
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)
    print(f'--- {ev} ---')
    print(f'GT: shape={gt_full.shape} 值分布 NF={np.sum(gt_full==0)} FO={np.sum(gt_full==1)} FU={np.sum(gt_full==2)}')
    best_shift_tot = {}
    n_checked = 0
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
        mask = config_f(sar)
        if mask.sum() < 50:
            continue
        # 位移扫描：F1 随 (dy,dx) 变化
        best = (0, 0, f1((mask & (g == 1)).sum(), (mask & (g != 1)).sum(), (~mask & (g == 1)).sum()))
        for dy in range(-shift_range, shift_range + 1):
            for dx in range(-shift_range, shift_range + 1):
                y1, x1 = y0 + dy, x0 + dx
                if y1 < 0 or x1 < 0 or y1 + 256 > gt_full.shape[0] or x1 + 256 > gt_full.shape[1]:
                    continue
                gs = gt_full[y1:y1 + 256, x1:x1 + 256]
                tp = (mask & (gs == 1)).sum()
                fp = (mask & (gs != 1)).sum()
                fn = (~mask & (gs == 1)).sum()
                fi = f1(tp, fp, fn)
                if fi > best[2]:
                    best = (dy, dx, fi)
        base = f1((mask & (g == 1)).sum(), (mask & (g != 1)).sum(), (~mask & (g == 1)).sum())
        if best[2] > base + 1e-9:
            best_shift_tot.setdefault((best[0], best[1]), 0)
            best_shift_tot[(best[0], best[1])] += 1
        n_checked += 1
        if n_checked >= n_patches:
            break
    print(f'检查 {n_checked} 个 patch：其中最佳位移非 (0,0) 的分布 = {best_shift_tot or "无（全部在(0,0)最优）"}')


for ev in ['20210727_Weihui', '20230609_NovaKakhovka']:
    check_event(ev)
