"""测试完整 GE 路由：scene 分类 → urban→urban_coh, open→int。"""
import sys, os
import numpy as np
import rasterio
from flood_strategies import all_strategies
from scene_classifier import classify_scene


def urban_coh(sar, pre_thr=0.5, delta_min=0.2):
    coh = all_strategies(sar)['coh']
    return coh & (sar[0] > pre_thr) & (sar[0] - sar[2] > delta_min)


def routed(sar):
    sc = classify_scene(sar)['scene']
    s = all_strategies(sar)
    if sc == 'urban':
        return urban_coh(sar), 'urban_coh'
    return s['int'], 'int'


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
    acc_o = [0, 0, 0]
    acc_u = [0, 0, 0]
    acc_a = [0, 0, 0]
    route_count = {}
    with rasterio.open(sar_path) as ds:
        for y0, x0 in positions:
            sar = ds.read(window=((y0, y0 + TILE), (x0, x0 + TILE)))
            gt = gt_full[y0:y0 + TILE, x0:x0 + TILE]
            m, chosen = routed(sar)
            route_count[chosen] = route_count.get(chosen, 0) + 1
            for acc, gbin in [(acc_o, gt == 1), (acc_u, gt == 2), (acc_a, gt > 0)]:
                acc[0] += (m & gbin).sum()
                acc[1] += (m & ~gbin).sum()
                acc[2] += (~m & gbin).sum()

    def rep(name, acc):
        tp, fp, fn = acc
        p_ = tp / (tp + fp) if tp + fp else 0
        r_ = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
        return f'{name}: F1={f1:.3f} P={p_:.3f} R={r_:.3f}'

    print(f'=== {event} (tile {len(positions)}, 路由分布 {route_count}) ===')
    print(f'  {rep("open", acc_o)}')
    print(f'  {rep("urban", acc_u)}')
    print(f'  {rep("all", acc_a)}')
