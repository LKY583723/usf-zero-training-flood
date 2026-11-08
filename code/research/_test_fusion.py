"""测试策略融合：int ∪ urban_coh 等，评估 open/urban/all F1（Weihui）。"""
import sys, os
import numpy as np
import rasterio
from flood_strategies import all_strategies

event = '20210727_Weihui'
data_dir = 'D:/DisasterM3/UrbanSARFloods'
sub = os.path.join(event, 'testing_case_orig', event)
with rasterio.open(os.path.join(data_dir, sub, f'{event}_GT.tif')) as ds:
    gt_full = ds.read(1)
TILE = 512
positions = [(y0, x0) for y0 in range(0, gt_full.shape[0] - TILE + 1, TILE)
             for x0 in range(0, gt_full.shape[1] - TILE + 1, TILE)
             if ((gt_full[y0:y0 + TILE, x0:x0 + TILE] > 0).sum()) >= 100]

fusions = {
    'int_only': lambda s: s['int'],
    'int_union_urban': lambda s: s['int'] | s['urban'],
    'int_cap_chg': lambda s: s['int'] & s['chg'],
    '(int_cap_chg)_union_urban': lambda s: (s['int'] & s['chg']) | s['urban'],
    'int_union_chg_union_urban': lambda s: s['int'] | s['chg'] | s['urban'],
}
acc = {k: {'open': [0,0,0], 'urban': [0,0,0], 'all': [0,0,0]} for k in fusions}
with rasterio.open(os.path.join(data_dir, sub, f'{event}_SAR.tif')) as ds:
    for y0, x0 in positions:
        sar = ds.read(window=((y0, y0 + TILE), (x0, x0 + TILE)))
        gt = gt_full[y0:y0 + TILE, x0:x0 + TILE]
        s = all_strategies(sar)
        for name, fn in fusions.items():
            m = fn(s)
            for cls, gbin in (('open', gt == 1), ('urban', gt == 2), ('all', gt > 0)):
                acc[name][cls][0] += (m & gbin).sum()
                acc[name][cls][1] += (m & ~gbin).sum()
                acc[name][cls][2] += (~m & gbin).sum()

print(f'Weihui 融合 (像素聚合, {len(positions)} tile):')
print(f'{"策略":28s} {"openF1":>8s} {"urbanF1":>8s} {"allF1":>8s}')
for name, a in acc.items():
    def f1(cls):
        tp, fp, fn = a[cls]
        p_ = tp / (tp + fp) if tp + fp else 0
        r_ = tp / (tp + fn) if tp + fn else 0
        return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
    print(f'{name:28s} {f1("open"):8.3f} {f1("urban"):8.3f} {f1("all"):8.3f}')
