"""诊断 Jubba_2：各策略 P/R + 真洪水vs假阳特征。"""
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

# 洪水 patch 列表 + 各自洪水像素
flood_patches = []
for f in files:
    r, c = parse_id(f)
    y0, x0 = r * 256, c * 256
    if y0 + 256 > gt_full.shape[0] or x0 + 256 > gt_full.shape[1]:
        continue
    g = gt_full[y0:y0 + 256, x0:x0 + 256]
    nf = (g > 0).sum()
    if nf >= 100:
        flood_patches.append((f, g, nf))
print(f'Jubba_2 洪水 patch: {len(flood_patches)}')
for f, g, nf in flood_patches[:25]:
    print(f'  {os.path.basename(f)}: 洪水 {nf} px ({nf/65536*100:.1f}%)')

# 各策略 P/R（无连通域）
strategies = ['int', 'chg', 'coh', 'int&chg']
acc = {s: [0, 0, 0] for s in strategies}
for f, g, nf in flood_patches:
    with rasterio.open(f) as ds:
        sar = ds.read().astype(np.float32)
    s = all_strategies(sar)
    masks = {'int': s['int'], 'chg': s['chg'], 'coh': s['coh'], 'int&chg': s['int'] & s['chg']}
    gb = g > 0
    for name, m in masks.items():
        acc[name][0] += (m & gb).sum()
        acc[name][1] += (m & ~gb).sum()
        acc[name][2] += (~m & gb).sum()

print('\n各策略（无连通域）:')
for name, (tp, fp, fn) in acc.items():
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
    print(f'  {name}: F1={f1:.3f} P={p_:.3f} R={r_:.3f} (tp={tp})')

# 真洪水 vs 假阳特征（int&chg）
feats = {k: {'true': [], 'fp': []} for k in ['post_vv', 'post_vh', 'pre_coh', 'texture']}
for f, g, nf in flood_patches:
    with rasterio.open(f) as ds:
        sar = ds.read().astype(np.float32)
    s = all_strategies(sar)
    pred = s['int'] & s['chg']
    true = pred & (g == 1)
    fp = pred & (g == 0)
    d = {'post_vv': sar[7], 'post_vh': sar[6], 'pre_coh': sar[0],
         'texture': np.abs(sar[7] - np.nanmean(sar[7]))}
    for k, arr in d.items():
        if arr[true].size:
            feats[k]['true'].append(arr[true])
        if arr[fp].size:
            feats[k]['fp'].append(arr[fp])
print('\n真洪水 vs 假阳 特征中位数:')
for k in feats:
    t = np.nanmedian(np.concatenate(feats[k]['true'])) if feats[k]['true'] else float('nan')
    fp_ = np.nanmedian(np.concatenate(feats[k]['fp'])) if feats[k]['fp'] else float('nan')
    print(f'  {k}: 真={t:.3f} 假阳={fp_:.3f}')
