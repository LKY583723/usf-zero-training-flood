"""高效检查覆盖区洪水 patch 的 GRD 有效景数（每 tif 打开一次，window 读）。"""
import glob
import re
import sys
import numpy as np
import rasterio

ALIGNED = 'D:/DisasterM3/UrbanSARFloods/timeseries/20231201_Jubba_1_aligned'
GT = 'D:/DisasterM3/UrbanSARFloods/20231201_Jubba_1/testing_case_orig/20231201_Jubba_1/20231201_Jubba_1_GT.tif'

with rasterio.open(GT) as ds:
    gt = ds.read(1)
    left = ds.bounds.left
    right = ds.bounds.right
    W = gt.shape[1]

files = sorted(glob.glob('D:/DisasterM3/UrbanSARFloods/testing_case_256/20231201_Jubba_1/*_SAR.tif'))
vv_files = sorted(glob.glob(f'{ALIGNED}/*_VV.tif'))

# 收集洪水 patch 的窗口
patch_windows = []  # (y0, x0, lon)
for f in files:
    m = re.search(r'ID_(\d+)_(\d+)_SAR', f)
    r, c = int(m.group(1)), int(m.group(2))
    y0, x0 = r * 256, c * 256
    g = gt[y0:y0 + 256, x0:x0 + 256]
    if (g > 0).sum() < 100:
        continue
    lon = left + (x0 + 128) / W * (right - left)
    patch_windows.append((y0, x0, lon))

print(f'洪水 patch: {len(patch_windows)}', flush=True)

# 每 tif 打开一次，读取所有 patch 窗口的中位数
n_valid = np.zeros(len(patch_windows), dtype=int)
for vf in vv_files:
    with rasterio.open(vf) as ds:
        for i, (y0, x0, lon) in enumerate(patch_windows):
            med = float(np.nanmedian(ds.read(1, window=((y0, y0 + 256), (x0, x0 + 256)))))
            if med < -5:
                n_valid[i] += 1

for (y0, x0, lon), n in sorted(zip(patch_windows, n_valid), key=lambda t: t[0][2]):
    mark = 'OK' if n >= 8 else ('~' if n >= 4 else 'X')
    print(f'  lon={lon:.2f}: {n}/12 有效 {mark}', flush=True)
print(f'有效景>=8: {(n_valid >= 8).sum()}/{len(patch_windows)}', flush=True)
