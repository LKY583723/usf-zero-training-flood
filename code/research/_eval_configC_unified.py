"""统一评测 config A/C/F（确定性路由）· 4 事件 × testing_case_256 patch × 像素聚合。

口径与 _eval_unified.py 完全一致（主表来源），补充 config C：
  A = S_int
  C = 确定性场景分类(classify_scene) → 决策表路由(route) → 最优单策略
  F = int∩chg + post_vv<−12 + 连通域(1500)
所有配置无 VLM（确定性），与报告主表同口径。
"""
import glob
import os
import re
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import rasterio
from flood_strategies import all_strategies, conn_filter
from scene_classifier import classify_scene
from flood_router import route

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256'
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
CONN_MIN = 500
POST_THR = -12.0


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def masks(sar):
    s = all_strategies(sar)
    # A
    a = s['int']
    # C: 确定性场景 → 路由
    scene = classify_scene(sar)
    labels = {"scene": scene.get("scene"), "scene_density": scene.get("scene_density")}
    routing = route(labels)
    top = routing["expectedBest"] or "int"
    c = s[top]
    # F
    f = s['int'] & s['chg']
    f = f & (np.nan_to_num(sar[7]) < POST_THR)
    f = conn_filter(f, CONN_MIN)
    return {'A': a, 'C': c, 'F': f}, routing


print('事件, 方法, openF1, urbanF1, allF1, (routed=top_strategy统计)')
for ev in EVENTS:
    patch_dir = os.path.join(PATCHES, ev)
    files = sorted(glob.glob(os.path.join(patch_dir, '*_SAR.tif')))
    if not files:
        print(f'{ev}: 无 patches，跳过')
        continue
    gt_path = os.path.join(DATA, ev, 'testing_case_orig', ev, f'{ev}_GT.tif')
    if not os.path.exists(gt_path):
        print(f'{ev}: 无 GT，跳过')
        continue
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)

    acc = {m: {'open': [0, 0, 0], 'urban': [0, 0, 0], 'all': [0, 0, 0]}
           for m in ['A', 'C', 'F']}
    route_counter = {}
    n_patch = 0
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
        n_patch += 1
        preds, routing = masks(sar)
        route_counter[routing['expectedBest']] = route_counter.get(routing['expectedBest'], 0) + 1
        for m, pred in preds.items():
            for cls, gbin in (('open', g == 1), ('urban', g == 2), ('all', g > 0)):
                acc[m][cls][0] += (pred & gbin).sum()
                acc[m][cls][1] += (pred & ~gbin).sum()
                acc[m][cls][2] += (~pred & gbin).sum()

    for m in ['A', 'C', 'F']:
        row = [ev, m]
        for cls in ['open', 'urban', 'all']:
            tp, fp, fn = acc[m][cls]
            p_ = tp / (tp + fp) if tp + fp else 0
            r_ = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
            row.append(f'{f1:.3f}')
        print(', '.join(row), f'  (patch={n_patch})')
    print(f'  [路由分布 {ev}]: {route_counter}')
