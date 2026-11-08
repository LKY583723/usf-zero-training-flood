"""场景自适应连通域阈值：urban tile 大阈值，open tile 中阈值。"""
import glob
import os
import re
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import rasterio
from flood_strategies import all_strategies, conn_filter
from scene_classifier import classify_scene

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256'
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def adaptive(sar, urban_min=2000, open_min=800):
    s = all_strategies(sar)
    base = s['int'] & s['chg']
    sc = classify_scene(sar)['scene']
    ms = urban_min if sc == 'urban' else open_min
    return conn_filter(base, ms)


methods = {
    'global1500': lambda sar: conn_filter(all_strategies(sar)['int'] & all_strategies(sar)['chg'], 1500),
    'adaptive': lambda sar: adaptive(sar),
}

print('事件, 方法, openF1, allF1')
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
    acc = {k: {'open': [0,0,0], 'all': [0,0,0]} for k in methods}
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
        for name, fn in methods.items():
            m = fn(sar)
            for cls, gbin in (('open', g == 1), ('all', g > 0)):
                acc[name][cls][0] += (m & gbin).sum()
                acc[name][cls][1] += (m & ~gbin).sum()
                acc[name][cls][2] += (~m & gbin).sum()
    for name, a in acc.items():
        row = [ev, name]
        for cls in ['open', 'all']:
            tp, fp, fn = a[cls]
            p_ = tp/(tp+fp) if tp+fp else 0
            r_ = tp/(tp+fn) if tp+fn else 0
            f1 = 2*p_*r_/(p_+r_) if p_+r_ else 0
            row.append(f'{f1:.3f}')
        print(', '.join(row))
