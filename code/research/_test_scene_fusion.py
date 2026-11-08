"""场景自适应融合：urban tile → urban_coh ∪ (int∩chg)，open tile → int∩chg。"""
import sys, os
import numpy as np
import rasterio
from flood_strategies import all_strategies
from scene_classifier import classify_scene


def fused(sar):
    sc = classify_scene(sar)['scene']
    s = all_strategies(sar)
    base = s['int'] & s['chg']
    if sc == 'urban':
        return base | s['urban'], 'urban', 'base∪urban'
    return base, sc, 'base'


for event in ['20210727_Weihui', '20230609_NovaKakhovka']:
    data_dir = 'D:/DisasterM3/UrbanSARFloods'
    sub = os.path.join(event, 'testing_case_orig', event)
    sar_path = os.path.join(data_dir, sub, f'{event}_SAR.tif')
    gt_path = os.path.join(data_dir, sub, f'{event}_GT.tif')
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)
    TILE = 512
    positions = [(y0, x0) for y0 in range(0, gt_full.shape[0] - TILE + 1, TILE)
                 for x0 in range(0, gt_full.shape[1] - TILE + 1, TILE)
                 if ((gt_full[y0:y0 + TILE, x0:x0 + TILE] > 0).sum()) >= 100]
    acc = {'open': [0,0,0], 'urban': [0,0,0], 'all': [0,0,0]}
    route = {}
    with rasterio.open(sar_path) as ds:
        for y0, x0 in positions:
            sar = ds.read(window=((y0, y0 + TILE), (x0, x0 + TILE)))
            gt = gt_full[y0:y0 + TILE, x0:x0 + TILE]
            m, sc, used = fused(sar)
            route[used] = route.get(used, 0) + 1
            for cls, gbin in (('open', gt == 1), ('urban', gt == 2), ('all', gt > 0)):
                acc[cls][0] += (m & gbin).sum()
                acc[cls][1] += (m & ~gbin).sum()
                acc[cls][2] += (~m & gbin).sum()

    def f1(cls):
        tp, fp, fn = acc[cls]
        p_ = tp / (tp + fp) if tp + fp else 0
        r_ = tp / (tp + fn) if tp + fn else 0
        return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0

    print(f'=== {event} (路由 {route}) ===')
    print(f'  open F1={f1("open"):.3f} urban F1={f1("urban"):.3f} all F1={f1("all"):.3f}')
