"""连通域过滤：int∩chg 基础上砍掉孤立小假阳（纯确定性，快）。"""
import sys, os
import numpy as np
import rasterio
import cv2
from flood_strategies import all_strategies


def conn_filter(mask, min_size):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[lab == i] = True
    return out


for event in ['20210727_Weihui', '20230609_NovaKakhovka']:
    data_dir = 'D:/DisasterM3/UrbanSARFloods'
    sub = os.path.join(event, 'testing_case_orig', event)
    with rasterio.open(os.path.join(data_dir, sub, f'{event}_GT.tif')) as ds:
        gt_full = ds.read(1)
    TILE = 512
    positions = [(y0, x0) for y0 in range(0, gt_full.shape[0]-TILE+1, TILE)
                 for x0 in range(0, gt_full.shape[1]-TILE+1, TILE)
                 if ((gt_full[y0:y0+TILE, x0:x0+TILE] > 0).sum()) >= 100]

    variants = {
        'int_cap_chg': lambda m: m,
        '+conn50': lambda m: conn_filter(m, 50),
        '+conn100': lambda m: conn_filter(m, 100),
        '+conn200': lambda m: conn_filter(m, 200),
    }
    acc = {k: [0,0,0] for k in variants}
    with rasterio.open(os.path.join(data_dir, sub, f'{event}_SAR.tif')) as ds:
        for y0, x0 in positions:
            sar = ds.read(window=((y0, y0+TILE), (x0, x0+TILE)))
            gt = gt_full[y0:y0+TILE, x0:x0+TILE] > 0
            s = all_strategies(sar)
            base = s['int'] & s['chg']
            for name, fn in variants.items():
                m = fn(base)
                acc[name][0] += (m & gt).sum()
                acc[name][1] += (m & ~gt).sum()
                acc[name][2] += (~m & gt).sum()

    print(f'=== {event} ===')
    for name, (tp, fp, fn) in acc.items():
        p_ = tp/(tp+fp) if tp+fp else 0
        r_ = tp/(tp+fn) if tp+fn else 0
        f1 = 2*p_*r_/(p_+r_) if p_+r_ else 0
        print(f'  {name}: F1={f1:.3f} P={p_:.3f} R={r_:.3f}')
