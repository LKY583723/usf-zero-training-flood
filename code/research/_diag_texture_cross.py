"""跨事件诊断：int∩chg 候选内，真洪水 vs 假阳的暗度/对比度分离。

特征：
  post_vv    灾后 VV（越暗越像水）
  contrast   |post_vv − patch均值|（离场景均值越远）
  local_tex  post_vv 3x3 局部标准差（真实纹理）
  vh_minus   post_vh − post_vv（极化特征）
若某个特征在 4 事件上都一致分离真/假，就是可泛化的物理线索。
"""
import glob
import os
import re
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')
import numpy as np
import rasterio
from flood_strategies import all_strategies

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256'
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def local_std(x, k=3):
    from scipy import ndimage
    return ndimage.uniform_filter((x - ndimage.uniform_filter(x, k)) ** 2, k) ** 0.5


print('事件, 组, n, post_vv, contrast, local_tex, vh_minus_vv, coh_chg')
for ev in EVENTS:
    patch_dir = os.path.join(PATCHES, ev)
    files = sorted(glob.glob(os.path.join(patch_dir, '*_SAR.tif')))
    gt_path = os.path.join(DATA, ev, 'testing_case_orig', ev, f'{ev}_GT.tif')
    if not (files and os.path.exists(gt_path)):
        continue
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)

    feats = {'true': {k: [] for k in ['post_vv', 'contrast', 'local_tex', 'vh_minus_vv', 'coh_chg']},
             'fp': {k: [] for k in ['post_vv', 'contrast', 'local_tex', 'vh_minus_vv', 'coh_chg']}}
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
        cand = s['int'] & s['chg']
        true = cand & (g == 1)
        fp = cand & (g == 0)
        post = sar[7]
        m = np.nanmean(post)
        d = {
            'post_vv': post,
            'contrast': np.abs(post - m),
            'local_tex': local_std(post),
            'vh_minus_vv': sar[6] - post,
            'coh_chg': sar[0] - sar[2],
        }
        for grp, mask in (('true', true), ('fp', fp)):
            if mask.sum() < 5:
                continue
            for k, arr in d.items():
                feats[grp][k].append(arr[mask])

    for grp in ('true', 'fp'):
        cells = [ev, grp, str(sum(len(x) for x in feats[grp]['post_vv']))]
        for k in ['post_vv', 'contrast', 'local_tex', 'vh_minus_vv', 'coh_chg']:
            v = np.concatenate(feats[grp][k]) if feats[grp][k] else np.array([np.nan])
            cells.append(f'{np.nanmedian(v):.3f}')
        print(', '.join(cells))
