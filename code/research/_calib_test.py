"""校准 GRD 幅度 → σ0 dB，验证与 USF post 量纲一致。"""
import re
import zipfile
import numpy as np
import rasterio

z = 'D:/DisasterM3/UrbanSARFloods/timeseries/20231201_Jubba_1/20231201_Jubba_1_2023-05-30.zip'
GT = 'D:/DisasterM3/UrbanSARFloods/20231201_Jubba_1/testing_case_orig/20231201_Jubba_1/20231201_Jubba_1_GT.tif'
patch_f = 'D:/DisasterM3/UrbanSARFloods/testing_case_256/20231201_Jubba_1/20231201_Jubba_1_ID_10_50_SAR.tif'


def parse_calibration(zpath, pol):
    with zipfile.ZipFile(zpath) as zf:
        cal = [n for n in zf.namelist() if '/annotation/calibration/' in n and pol in n][0]
        xml = zf.read(cal).decode('utf-8', errors='ignore')
    vecs = re.findall(r'<calibrationVector>(.*?)</calibrationVector>', xml, re.S)
    out = []
    for v in vecs:
        line = float(re.search(r'<line>(\S+)</line>', v).group(1))
        pix = np.array([float(x) for x in re.search(r'<pixel[^>]*>(.*?)</pixel>', v, re.S).group(1).split()])
        sn = np.array([float(x) for x in re.search(r'<sigmaNought[^>]*>(.*?)</sigmaNought>', v, re.S).group(1).split()])
        out.append((line, pix, sn))
    return out


def calibrate_patch(amp, y0, calib, max_col):
    """amp: 2D patch（原始幅度），y0: patch 全局起始行。返回 σ0 dB。"""
    out = np.zeros_like(amp, dtype=np.float32)
    lines = np.array([c[0] for c in calib])
    for y in range(amp.shape[0]):
        gy = y0 + y
        # 找最近 vector line
        di = np.argmin(np.abs(lines - gy))
        pix, sn = calib[di][1], calib[di][2]
        for x in range(amp.shape[1]):
            gx = x  # pixel 在 GRD 内是全局列（patch 假设从 0 起）
            if gx > max_col:
                out[y, x] = 0
                continue
            # 找最近 pixel
            pi = np.argmin(np.abs(pix - gx))
            sn_val = sn[pi]
            dn = amp[y, x]
            out[y, x] = 10 * np.log10(max(dn * dn / (sn_val * sn_val), 1e-12))
    return out


# 读 patch 的原始 GRD 幅度（从 zip）
with zipfile.ZipFile(z) as zf:
    tif = [n for n in zf.namelist() if '/measurement/' in n and 'vv' in n and n.endswith('.tiff')][0]
# patch ID_10_50 → 整景 (10*256, 50*256) = (2560, 12800)
y0, x0 = 2560, 12800
src = f'/vsizip/{z}/{tif}'
with rasterio.open(src) as ds:
    amp = ds.read(1, window=((y0, y0 + 256), (x0, x0 + 256)))

calib = parse_calibration(z, 'vv')
db = calibrate_patch(amp, y0, calib, ds.width)

with rasterio.open(patch_f) as ds:
    sar = ds.read()
print('校准 σ0 dB 中位数: %.2f' % np.median(db))
print('USF patch post VV (band7): %.2f' % np.median(sar[7]))
print('USF patch pre VV (band5): %.2f' % np.median(sar[5]))
print('GRD 原始幅度中位数: %.0f' % np.median(amp))
