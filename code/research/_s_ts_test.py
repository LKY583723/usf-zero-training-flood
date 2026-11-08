"""S_ts 验证：GRD 12 景校准 σ0 → z-score 异常检测，对比 GE 单对。

在 GRD 覆盖区洪水 patch 上，对比：
- GE 单对（int∩chg+post+conn）
- S_ts（z-score：post 显著低于灾前均值）
"""
import glob
import os
import re
import sys
import zipfile
import numpy as np
import rasterio

from flood_strategies import all_strategies, conn_filter

TS = 'D:/DisasterM3/UrbanSARFloods/timeseries/20231201_Jubba_1'
GT = 'D:/DisasterM3/UrbanSARFloods/20231201_Jubba_1/testing_case_orig/20231201_Jubba_1/20231201_Jubba_1_GT.tif'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256/20231201_Jubba_1'
zips = sorted(glob.glob(os.path.join(TS, '*.zip')))


def parse_calibration(zpath, pol):
    with zipfile.ZipFile(zpath) as zf:
        cal = [n for n in zf.namelist() if '/annotation/calibration/' in n and pol in n][0]
        xml = zf.read(cal).decode('utf-8', errors='ignore')
    out = []
    for v in re.findall(r'<calibrationVector>(.*?)</calibrationVector>', xml, re.S):
        line = float(re.search(r'<line>(\S+)</line>', v).group(1))
        pix = np.array([float(x) for x in re.search(r'<pixel[^>]*>(.*?)</pixel>', v, re.S).group(1).split()])
        sn = np.array([float(x) for x in re.search(r'<sigmaNought[^>]*>(.*?)</sigmaNought>', v, re.S).group(1).split()])
        out.append((line, pix, sn))
    return out


def grd_patch_sigma0(zpath, pol, y0, x0, win=256):
    """返回 patch 的 σ0 dB（校准）。"""
    with zipfile.ZipFile(zpath) as zf:
        tif = [n for n in zf.namelist() if f'/measurement/' in n and pol in n and n.endswith('.tiff')][0]
        src = f'/vsizip/{zpath}/{tif}'
        with rasterio.open(src) as ds:
            amp = ds.read(1, window=((y0, y0 + win), (x0, x0 + win)))
            W = ds.width
    calib = parse_calibration(zpath, pol)
    lines = np.array([c[0] for c in calib])
    out = np.zeros_like(amp, dtype=np.float32)
    for y in range(win):
        gy = y0 + y
        di = int(np.argmin(np.abs(lines - gy)))
        pix, sn = calib[di][1], calib[di][2]
        # 对 patch 每列，从 pix 插值 sn
        cols = x0 + np.arange(win)
        cols_c = np.clip(cols, 0, pix[-1])
        sn_col = np.interp(cols_c, pix, sn)
        dn = amp[y].astype(np.float32)
        with np.errstate(divide='ignore'):
            out[y] = 10 * np.log10(np.maximum(dn * dn / (sn_col * sn_col), 1e-12))
    return out


# 洪水 patch
with rasterio.open(GT) as ds:
    gt = ds.read(1)
    left, right = ds.bounds.left, ds.bounds.right
    W = gt.shape[1]
files = sorted(glob.glob(os.path.join(PATCHES, '*_SAR.tif')))
coverage = []
for f in files:
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(f))
    r, c = int(m.group(1)), int(m.group(2))
    y0, x0 = r * 256, c * 256
    g = gt[y0:y0 + 256, x0:x0 + 256]
    if (g > 0).sum() < 100:
        continue
    lon = left + (x0 + 128) / W * (right - left)
    if lon >= 42.7:  # GRD 覆盖区
        coverage.append((f, y0, x0, g, lon))
print(f'覆盖区洪水 patch: {len(coverage)}', flush=True)

# 测试前 15 个 patch（控制时间）
results_ge = []
results_ts = []
for i, (f, y0, x0, g, lon) in enumerate(coverage[:15]):
    # GE 单对
    with rasterio.open(f) as ds:
        sar = ds.read().astype(np.float32)
    s = all_strategies(sar)
    base = s['int'] & s['chg']
    base = base & (np.nan_to_num(sar[7]) < -12)
    ge_mask = conn_filter(base, 1500)
    # S_ts: 12 景 σ0 + post
    ts_vv = []
    for zp in zips:
        db = grd_patch_sigma0(zp, 'vv', y0, x0)
        ts_vv.append(db)
    ts = np.stack(ts_vv)  # (12, 256, 256)
    mu = np.nanmean(ts, axis=0)
    sd = np.nanstd(ts, axis=0)
    post = sar[7]  # USF post VV
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (post - mu) / np.maximum(sd, 0.5)
    ts_mask = z < -2  # post 显著低于灾前
    ts_mask = ts_mask & np.isfinite(post)
    # F1
    def f1m(mask, gt_b):
        tp = (mask & gt_b).sum(); fp = (mask & ~gt_b).sum(); fn = (~mask & gt_b).sum()
        p_ = tp/(tp+fp) if tp+fp else 0; r_ = tp/(tp+fn) if tp+fn else 0
        return 2*p_*r_/(p_+r_) if p_+r_ else 0
    gb = g > 0
    f_ge = f1m(ge_mask, gb)
    f_ts = f1m(ts_mask, gb)
    results_ge.append(f_ge)
    results_ts.append(f_ts)
    print(f'patch({y0},{x0}) lon={lon:.2f}: GE={f_ge:.3f} S_ts={f_ts:.3f}', flush=True)

print(f'\n平均: GE单对={np.mean(results_ge):.3f} S_ts={np.mean(results_ts):.3f}', flush=True)
