"""诊断 NovaKakhovka：各策略 open F1（patch 级，无连通域）。"""
import glob
import os
import re
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import rasterio
from flood_strategies import all_strategies

ev = '20230609_NovaKakhovka'
PATCHES = f'D:/DisasterM3/UrbanSARFloods/testing_case_256/{ev}'
GT_PATH = f'D:/DisasterM3/UrbanSARFloods/{ev}/testing_case_orig/{ev}/{ev}_GT.tif'


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


with rasterio.open(GT_PATH) as ds:
    gt_full = ds.read(1)

files = sorted(glob.glob(os.path.join(PATCHES, '*_SAR.tif')))
strategies = ['int', 'chg', 'coh', 'urban',
              'int&chg', 'int|chg', 'int|urban', 'int&chg|urban']
acc = {s: {'open': [0,0,0], 'all': [0,0,0]} for s in strategies}
n = 0
for f in files:
    r, c = parse_id(f)
    y0, x0 = r*256, c*256
    if y0+256 > gt_full.shape[0] or x0+256 > gt_full.shape[1]:
        continue
    g = gt_full[y0:y0+256, x0:x0+256]
    if (g > 0).sum() < 100:
        continue
    with rasterio.open(f) as ds:
        sar = ds.read().astype(np.float32)
    s = all_strategies(sar)
    masks = {
        'int': s['int'], 'chg': s['chg'], 'coh': s['coh'], 'urban': s['urban'],
        'int&chg': s['int'] & s['chg'], 'int|chg': s['int'] | s['chg'],
        'int|urban': s['int'] | s['urban'], 'int&chg|urban': (s['int'] & s['chg']) | s['urban'],
    }
    n += 1
    for name, m in masks.items():
        for cls, gbin in (('open', g==1), ('all', g>0)):
            acc[name][cls][0] += (m & gbin).sum()
            acc[name][cls][1] += (m & ~gbin).sum()
            acc[name][cls][2] += (~m & gbin).sum()

print(f'NovaKakhovka (洪水 patch={n})')
print(f'{"策略":12s} {"openF1":>8s} {"P":>7s} {"R":>7s} {"allF1":>8s}')
for name, a in acc.items():
    row = [name]
    for cls in ['open', 'all']:
        tp, fp, fn = a[cls]
        p_ = tp/(tp+fp) if tp+fp else 0
        r_ = tp/(tp+fn) if tp+fn else 0
        f1 = 2*p_*r_/(p_+r_) if p_+r_ else 0
        row.append(f'{f1:.3f}')
    tp, fp, fn = a['open']
    p_ = tp/(tp+fp) if tp+fp else 0
    r_ = tp/(tp+fn) if tp+fn else 0
    print(f'{name:12s} {row[1]:>8s} {p_:7.3f} {r_:7.3f} {row[2]:>8s}')
