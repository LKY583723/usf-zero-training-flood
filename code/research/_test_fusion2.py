"""融合测试：int∩chg 基础 + urban_coh 调优参数，跨事件验证。"""
import sys, os
import numpy as np
import rasterio
from flood_strategies import all_strategies, CO_PRE_VH, CO_CO_VH, INT_POST_VV


def urban_tuned(sar, pre_thr=0.6, delta_min=0.25, post_min=-6.0):
    pre = sar[CO_PRE_VH]
    co = sar[CO_CO_VH]
    delta = np.nan_to_num(pre) - np.nan_to_num(co)
    m = (delta > delta_min) & (pre > pre_thr)
    if post_min is not None:
        m = m & (np.nan_to_num(sar[INT_POST_VV]) > post_min)
    return m


for event in ['20210727_Weihui', '20230609_NovaKakhovka', '20231201_Jubba_1']:
    data_dir = 'D:/DisasterM3/UrbanSARFloods'
    sub = os.path.join(event, 'testing_case_orig', event)
    sar_path = os.path.join(data_dir, sub, f'{event}_SAR.tif')
    if not os.path.exists(sar_path):
        sar_path = os.path.join(data_dir, sub, '20231201_Jubba_SAR.tif')
    gt_path = os.path.join(data_dir, sub, f'{event}_GT.tif')
    if not os.path.exists(gt_path):
        print(f'{event}: 缺数据'); continue
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)
    TILE = 512
    positions = [(y0, x0) for y0 in range(0, gt_full.shape[0]-TILE+1, TILE)
                 for x0 in range(0, gt_full.shape[1]-TILE+1, TILE)
                 if ((gt_full[y0:y0+TILE, x0:x0+TILE] > 0).sum()) >= 100]

    names = ['int_cap_chg', '+urban_tuned']
    acc = {k: {'open': [0,0,0], 'urban': [0,0,0], 'all': [0,0,0]} for k in names}
    with rasterio.open(sar_path) as ds:
        for y0, x0 in positions:
            sar = ds.read(window=((y0, y0+TILE), (x0, x0+TILE)))
            gt = gt_full[y0:y0+TILE, x0:x0+TILE]
            s = all_strategies(sar)
            m_base = s['int'] & s['chg']
            masks = {'int_cap_chg': m_base, '+urban_tuned': m_base | urban_tuned(sar)}
            for name, m in masks.items():
                for cls, gbin in (('open', gt==1), ('urban', gt==2), ('all', gt>0)):
                    acc[name][cls][0] += (m & gbin).sum()
                    acc[name][cls][1] += (m & ~gbin).sum()
                    acc[name][cls][2] += (~m & gbin).sum()

    def f1(cls, a):
        tp, fp, fn = a[cls]
        p_ = tp/(tp+fp) if tp+fp else 0
        r_ = tp/(tp+fn) if tp+fn else 0
        return 2*p_*r_/(p_+r_) if p_+r_ else 0

    print(f'=== {event} ===')
    for name, a in acc.items():
        print(f'  {name}: open={f1("open",a):.3f} urban={f1("urban",a):.3f} all={f1("all",a):.3f}')
