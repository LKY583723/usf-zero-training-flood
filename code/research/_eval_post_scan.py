"""post_vv 暗水阈值扫描：int∩chg+conn + post<阈值，4 事件（patch 级）。"""
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
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
POST_THRESH = [None, -12, -13, -14, -15]


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


print('事件, post_thr, openF1, allF1')
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

    for pt in POST_THRESH:
        acc = {'open': [0,0,0], 'all': [0,0,0]}
        for s, sar, g in patches:
            base = s['int'] & s['chg']
            if pt is not None:
                base = base & (np.nan_to_num(sar[7]) < pt)
            m = conn_filter(base, 1500)
            for cls, gbin in (('open', g == 1), ('all', g > 0)):
                acc[cls][0] += (m & gbin).sum()
                acc[cls][1] += (m & ~gbin).sum()
                acc[cls][2] += (~m & gbin).sum()
        row = [ev, '无' if pt is None else str(pt)]
        for cls in ['open', 'all']:
            tp, fp, fn = acc[cls]
            p_ = tp/(tp+fp) if tp+fp else 0
            r_ = tp/(tp+fn) if tp+fn else 0
            f1 = 2*p_*r_/(p_+r_) if p_+r_ else 0
            row.append(f'{f1:.3f}')
        print(', '.join(row))
