"""GE config F 在 Houston（训练事件，GE 零训练未见）上的表现——验证规则泛化。"""
import glob
import os
import sys
import numpy as np
import rasterio
from flood_strategies import all_strategies, conn_filter

root = 'D:/DisasterM3/UrbanSARFloods/urban_sar_floods'
tiles = []
for cat in ['02_FO', '03_FU']:
    sar_files = glob.glob(f'{root}/{cat}/SAR/*Houston*_SAR.tif')
    for sf in sar_files:
        gt_dir = os.path.dirname(sf)
        gt_name = os.path.basename(sf).replace('_SAR.tif', '_GT.tif')
        gt_dir = gt_dir.replace('\\SAR', '\\GT').replace('/SAR', '/GT')
        gf = os.path.join(gt_dir, gt_name)
        if os.path.exists(gf):
            tiles.append((sf, gf, cat))
n_fo = sum(1 for t in tiles if t[2] == '02_FO')
n_fu = sum(1 for t in tiles if t[2] == '03_FU')
print(f'Houston 洪水 tile: {len(tiles)} (FO={n_fo}, FU={n_fu})')

acc = {'open': [0,0,0], 'urban': [0,0,0], 'all': [0,0,0]}
n = 0
for sf, gf, cat in tiles:
    with rasterio.open(sf) as ds:
        sar = ds.read().astype(np.float32)
    with rasterio.open(gf) as ds:
        gt = ds.read(1)
    if (gt > 0).sum() < 100:
        continue
    s = all_strategies(sar)
    base = s['int'] & s['chg']
    base = base & (np.nan_to_num(sar[7]) < -12)
    m = conn_filter(base, 1500)
    n += 1
    for cls, gbin in (('open', gt == 1), ('urban', gt == 2), ('all', gt > 0)):
        acc[cls][0] += (m & gbin).sum()
        acc[cls][1] += (m & ~gbin).sum()
        acc[cls][2] += (~m & gbin).sum()

print(f'有效洪水 tile: {n}')
for cls, (tp, fp, fn) in acc.items():
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
    print(f'  {cls}: F1={f1:.3f} P={p_:.3f} R={r_:.3f} (tp={tp})')
