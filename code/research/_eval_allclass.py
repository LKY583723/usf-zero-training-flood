"""对齐 BCP 口径：GE-F 的 all-class（GT>0，含城区）像素聚合 F1。

BCP 论文的 per-site F1 是全类别口径（含 urban）；此前我们对比用 open-only。
本脚本同时给 open 与 all 两个口径，验证 GE-F vs BCP 的结论是否与口径无关。
"""
import glob
import os
import re
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')
import numpy as np
import rasterio
from flood_strategies import all_strategies, conn_filter

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256'
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
POST_THR, CONN = -12.0, 500


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


# 每个场景: open=[tp,fp,fn]  all=[tp,fp,fn]
acc = {sc: {'open': [0, 0, 0], 'all': [0, 0, 0]} for sc in SCENES}
for sc in SCENES:
    files = sorted(glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')))
    gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)
    for f in files:
        r, c = parse_id(f)
        y0, x0 = r * 256, c * 256
        if y0 + 256 > gt_full.shape[0] or x0 + 256 > gt_full.shape[1]:
            continue
        gt = gt_full[y0:y0 + 256, x0:x0 + 256]
        if (gt > 0).sum() < 100:
            continue
        with rasterio.open(f) as ds:
            sar = ds.read().astype(np.float32)
        s = all_strategies(sar)
        m = conn_filter(s['int'] & s['chg'] & (np.nan_to_num(sar[7]) < POST_THR), CONN)
        a = acc[sc]
        a['open'][0] += (m & (gt == 1)).sum()
        a['open'][1] += (m & (gt != 1)).sum()
        a['open'][2] += (~m & (gt == 1)).sum()
        a['all'][0] += (m & (gt > 0)).sum()
        a['all'][1] += (m & (gt == 0)).sum()
        a['all'][2] += (~m & (gt > 0)).sum()
    print(f'  {sc}: open F1={f1(*acc[sc]["open"]):.3f}  all F1={f1(*acc[sc]["all"]):.3f}', flush=True)

# Jubba 合并 + 3 事件均值（两口径）
print('\n=== 3 事件协议（Jubba 合并）===')
print('场景, open F1, all F1, BCP(all)')
rows = []
for sc, bcp in [('20210727_Weihui', 0.75), ('20230609_NovaKakhovka', 0.49),
                ('Jubba(合并)', 0.68)]:
    if sc == 'Jubba(合并)':
        j_open = [0, 0, 0]; j_all = [0, 0, 0]
        for i in range(3):
            j_open[i] = acc['20231201_Jubba_1']['open'][i] + acc['20231201_Jubba_2']['open'][i]
            j_all[i] = acc['20231201_Jubba_1']['all'][i] + acc['20231201_Jubba_2']['all'][i]
        of, af = f1(*j_open), f1(*j_all)
    else:
        of, af = f1(*acc[sc]['open']), f1(*acc[sc]['all'])
    rows.append((sc, of, af, bcp))
    print(f'{sc}: open={of:.3f}  all={af:.3f}  BCP={bcp}')

mean_open = np.mean([r[1] for r in rows])
mean_all = np.mean([r[2] for r in rows])
print(f'\n3事件均值: open={mean_open:.3f}  all={mean_all:.3f}')
for sc, of, af, bcp in rows:
    print(f'  vs BCP: {sc}: all/BCP = {af:.3f}/{bcp:.3f} -> {"赢" if af >= bcp else "输"}')
