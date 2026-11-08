"""在 Jubba patches 上跑 GE（config A/C/F），聚合指标。

因为 Jubba 整景 SAR 在 hf 上 403 无法下载，用 testing_case_256 的 256² patch。
用法：python _ge_jubba_patches.py 20231201_Jubba_1 [configs...]
"""
import glob
import os
import re
import sys
import tempfile

import numpy as np
import rasterio

from flood_ge import process_tile  # noqa: E402
from flood_metrics import metrics  # noqa: E402

ev = sys.argv[1]
configs = sys.argv[2:] or ['A', 'C', 'F']
PATCH_DIR = f'D:/DisasterM3/UrbanSARFloods/testing_case_256/{ev}'
GT_PATH = f'D:/DisasterM3/UrbanSARFloods/{ev}/testing_case_orig/{ev}/{ev}_GT.tif'

with rasterio.open(GT_PATH) as ds:
    gt_full = ds.read(1)
print('整景 GT:', gt_full.shape, flush=True)

files = sorted(glob.glob(os.path.join(PATCH_DIR, '*_SAR.tif')))
print('patch 文件:', len(files), flush=True)


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


tmp = tempfile.mkdtemp()
# 像素聚合 tp/fp/fn
acc = {c: {'open': [0, 0, 0], 'urban': [0, 0, 0], 'all': [0, 0, 0]} for c in configs}
n_flood = 0
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
    n_flood += 1
    for cfg in configs:
        res = process_tile(sar, g, cfg, tmp, samples=1, no_vlm=False)
        pred = res['fused']
        for cls, gbin in (('open', g == 1), ('urban', g == 2), ('all', g > 0)):
            acc[cfg][cls][0] += (pred & gbin).sum()
            acc[cfg][cls][1] += (pred & ~gbin).sum()
            acc[cfg][cls][2] += (~pred & gbin).sum()

print(f'洪水 patch: {n_flood}', flush=True)
for cfg in configs:
    print(f'--- config {cfg} ---', flush=True)
    for cls, (tp, fp, fn) in acc[cfg].items():
        p_ = tp / (tp + fp) if tp + fp else 0
        r_ = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
        print(f'  {cls}: F1={f1:.3f} P={p_:.3f} R={r_:.3f} (tp={tp}, fp={fp}, fn={fn})', flush=True)
