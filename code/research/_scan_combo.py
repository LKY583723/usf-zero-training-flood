"""post × 纹理 组合扫描：找每事件最优（Weihui/Nova/Jubba_1）。"""
import glob
import os
import re
import sys
import numpy as np
import rasterio
from flood_strategies import all_strategies, conn_filter


def load_patches(event):
    PATCHES = f'D:/DisasterM3/UrbanSARFloods/testing_case_256/{event}'
    GT_PATH = f'D:/DisasterM3/UrbanSARFloods/{event}/testing_case_orig/{event}/{event}_GT.tif'
    if not os.path.exists(GT_PATH):
        return None

    def parse_id(fn):
        m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
        return int(m.group(1)), int(m.group(2))

    with rasterio.open(GT_PATH) as ds:
        gt_full = ds.read(1)
    files = sorted(glob.glob(os.path.join(PATCHES, '*_SAR.tif')))
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
    return patches


def eval_combo(patches, post_thr, tex_thr):
    acc = [0, 0, 0]
    for s, sar, g in patches:
        base = s['int'] & s['chg']
        base = base & (np.nan_to_num(sar[7]) < post_thr)
        if tex_thr:
            tex = np.abs(sar[7] - np.nanmean(sar[7]))
            base = base & (tex > tex_thr)
        m = conn_filter(base, 1500)
        gb = g > 0
        acc[0] += (m & gb).sum()
        acc[1] += (m & ~gb).sum()
        acc[2] += (~m & gb).sum()
    tp, fp, fn = acc
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


for event, posts, texs in [
    ('20210727_Weihui', [-11, -12, -13], [0, 2, 2.5, 3]),
    ('20230609_NovaKakhovka', [-12, -13, -14], [0, 2, 2.5]),
    ('20231201_Jubba_1', [-11, -12, -13], [0, 2]),
]:
    patches = load_patches(event)
    if not patches:
        continue
    print(f'=== {event} ===')
    results = []
    for pt in posts:
        for tex in texs:
            f1 = eval_combo(patches, pt, tex)
            results.append((f1, pt, tex))
    results.sort(reverse=True)
    print('  Top 5:')
    for f1, pt, tex in results[:5]:
        print(f'    post={pt} tex={tex}: F1={f1:.3f}')
