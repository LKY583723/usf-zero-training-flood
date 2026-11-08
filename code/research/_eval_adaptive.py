"""自适应暗水阈值验证：4 事件 × testing_case_256 patch × 像素聚合 open F1。

方案对比（候选 = int∩chg）：
  F_fixed    当前报告：post_vv < −12 + conn(1500)
  F_otsu     post_vv < Otsu(候选像素 post_vv) + conn(1500)      —— 逐场景自适应
  F_otsu500  F_otsu + conn(500)                                  —— 自适应 + 小连通域
  F_q30      post_vv < P30(候选 post_vv) + conn(500)             —— 自适应分位数
若 F_otsu 在 4 事件都 ≥ 当前 F_fixed，则"逐场景自适应阈值"是合法可泛化改进。
"""
import glob
import os
import re
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')
import numpy as np
import rasterio
import cv2
from flood_strategies import all_strategies, conn_filter

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256'
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def otsu_thresh(vals):
    """返回候选像素 post_vv 的 Otsu 分界阈值（dB）。"""
    v = np.asarray(vals, dtype=np.float32)
    v = v[np.isfinite(v)]
    if len(v) < 100:
        return -12.0
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        return -12.0
    q = np.clip((v - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    ret, _ = cv2.threshold(q, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret 是 0-255 归一化空间的阈值，映射回 dB
    return lo + (hi - lo) * ret / 255.0


def masks(sar, scheme):
    s = all_strategies(sar)
    base = s['int'] & s['chg']
    post = np.nan_to_num(sar[7])
    if scheme == 'fixed':
        return conn_filter(base & (post < -12.0), 1500)
    if scheme == 'otsu':
        thr = otsu_thresh(post[base])
        return conn_filter(base & (post < thr), 1500)
    if scheme == 'otsu500':
        thr = otsu_thresh(post[base])
        return conn_filter(base & (post < thr), 500)
    if scheme == 'q30':
        thr = np.nanpercentile(post[base], 30)
        return conn_filter(base & (post < thr), 500)
    raise ValueError(scheme)


print('事件, 方案, openF1, urbanF1, allF1')
for ev in EVENTS:
    patch_dir = os.path.join(PATCHES, ev)
    files = sorted(glob.glob(os.path.join(patch_dir, '*_SAR.tif')))
    gt_path = os.path.join(DATA, ev, 'testing_case_orig', ev, f'{ev}_GT.tif')
    if not (files and os.path.exists(gt_path)):
        continue
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)
    acc = {m: {'open': [0, 0, 0], 'urban': [0, 0, 0], 'all': [0, 0, 0]}
           for m in ['fixed', 'otsu', 'otsu500', 'q30']}
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
        for m in acc:
            pred = masks(sar, m)
            for cls, gbin in (('open', g == 1), ('urban', g == 2), ('all', g > 0)):
                acc[m][cls][0] += (pred & gbin).sum()
                acc[m][cls][1] += (pred & ~gbin).sum()
                acc[m][cls][2] += (~pred & gbin).sum()

    for m in ['fixed', 'otsu', 'otsu500', 'q30']:
        row = [ev, m]
        for cls in ['open', 'urban', 'all']:
            tp, fp, fn = acc[m][cls]
            p_ = tp / (tp + fp) if tp + fp else 0
            r_ = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
            row.append(f'{f1:.3f}')
        print(', '.join(row))
