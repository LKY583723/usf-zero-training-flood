"""urban_coh 参数扫描：城区 tile urban F1 + 整体 all F1（Weihui）。"""
import sys, os
import numpy as np
import rasterio
from flood_strategies import CO_PRE_VH, CO_CO_VH, INT_POST_VV

event = '20210727_Weihui'
data_dir = 'D:/DisasterM3/UrbanSARFloods'
sub = os.path.join(event, 'testing_case_orig', event)
with rasterio.open(os.path.join(data_dir, sub, f'{event}_GT.tif')) as ds:
    gt_full = ds.read(1)
TILE = 512
all_pos = [(y0, x0) for y0 in range(0, gt_full.shape[0] - TILE + 1, TILE)
           for x0 in range(0, gt_full.shape[1] - TILE + 1, TILE)
           if ((gt_full[y0:y0 + TILE, x0:x0 + TILE] > 0).sum()) >= 100]
urban_pos = [p for p in all_pos if (gt_full[p[0]:p[0]+TILE, p[1]:p[1]+TILE] == 2).sum() >= 100]
print(f'总 tile {len(all_pos)}, 城区 tile {len(urban_pos)}')


def urban_mask(sar, pre_thr, delta_min, post_min=None):
    pre = sar[CO_PRE_VH]
    co = sar[CO_CO_VH]
    delta = np.nan_to_num(pre) - np.nan_to_num(co)
    m = (delta > delta_min) & (pre > pre_thr)
    if post_min is not None:
        m = m & (np.nan_to_num(sar[INT_POST_VV]) > post_min)
    return m


def agg(positions, sar_ds, gt_full, fn):
    acc = {'open': [0,0,0], 'urban': [0,0,0], 'all': [0,0,0]}
    for y0, x0 in positions:
        sar = sar_ds.read(window=((y0, y0+TILE), (x0, x0+TILE)))
        gt = gt_full[y0:y0+TILE, x0:x0+TILE]
        m = fn(sar)
        for cls, gbin in (('open', gt==1), ('urban', gt==2), ('all', gt>0)):
            acc[cls][0] += (m & gbin).sum()
            acc[cls][1] += (m & ~gbin).sum()
            acc[cls][2] += (~m & gbin).sum()
    return acc


def f1(acc, cls):
    tp, fp, fn = acc[cls]
    p_ = tp / (tp+fp) if tp+fp else 0
    r_ = tp / (tp+fn) if tp+fn else 0
    return 2*p_*r_/(p_+r_) if p_+r_ else 0


with rasterio.open(os.path.join(data_dir, sub, f'{event}_SAR.tif')) as ds:
    print('扫描 urban_coh 参数（城区 tile urban F1 / 整体 all F1）:')
    print(f'{"pre_thr":>8s} {"delta":>6s} {"post_vv":>8s} {"城F1":>6s} {"城P":>6s} {"整F1":>6s}')
    for pre_thr in [0.4, 0.5, 0.6]:
        for delta_min in [0.15, 0.2, 0.25]:
            for post_min in [None, -6.0, -5.0]:
                fn = lambda s, pt=pre_thr, dm=delta_min, pm=post_min: urban_mask(s, pt, dm, pm)
                au = agg(urban_pos, ds, gt_full, fn)
                aa = agg(all_pos, ds, gt_full, fn)
                uf = f1(au, 'urban'); up = au['urban'][0]/max(au['urban'][0]+au['urban'][1],1)
                af = f1(aa, 'all')
                pm_str = '无' if post_min is None else f'{post_min}'
                print(f'{pre_thr:8.1f} {delta_min:6.2f} {pm_str:>8s} {uf:6.3f} {up:6.3f} {af:6.3f}')
