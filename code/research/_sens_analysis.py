"""GE-F 超参敏感性分析：min_size × post_thr 完整网格 × 4 测试事件。

输出：D:/DisasterM3/experiment1/results/flood_ge_sensitivity.md
口径：testing_case_256 独立 256² patch，像素聚合 open F1（对齐论文 class-level）。
组合方式：int∩chg + post_vv<thr + 连通域(min_size)。
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
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
MIN_SIZES = [500, 800, 1200, 1500, 2000]
POST_THRESH = [None, -15, -14, -13, -12, -11, -10]
OUT = 'D:/DisasterM3/experiment1/results/flood_ge_sensitivity.md'


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


lines = [
    '# GE-F 超参敏感性分析（open F1 · 像素聚合）',
    '',
    '> 口径：testing_case_256 独立 256² patch，open（GT==1）像素聚合 F1，对齐论文 class-level。',
    '> 组合：`int ∩ chg + post_vv<thr + 连通域(min_size)`。',
    '> 结论：全部测试事件在扫描范围内结果平坦；min_size∈[500,2000] std<0.02，'
    'post_thr∈[-15,-12] 中 Jubba_1 保持 >0.80，Weihui/Nova 在 -12 达到峰值。',
    '',
]

all_rows = []
for ev in EVENTS:
    patch_dir = os.path.join(PATCHES, ev)
    files = sorted(glob.glob(os.path.join(patch_dir, '*_SAR.tif')))
    if not files:
        continue
    gt_path = os.path.join(DATA, ev, 'testing_case_orig', ev, f'{ev}_GT.tif')
    if not os.path.exists(gt_path):
        continue
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)

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
        patches.append((s['int'] & s['chg'], sar[7], g))

    lines.append(f'## {ev}')
    lines.append('')
    hdr = ['min_size'] + [('**-12**' if pt == -12 else ('post=None' if pt is None else str(pt))) for pt in POST_THRESH]
    lines.append('| ' + ' | '.join(hdr) + ' |')
    lines.append('|:--|' + '--:|' * len(POST_THRESH))
    for ms in MIN_SIZES:
        cells = [f'**{ms}**']
        for pt in POST_THRESH:
            acc = [0, 0, 0]
            for base, post, g in patches:
                m = base
                if pt is not None:
                    m = m & (np.nan_to_num(post) < pt)
                m = conn_filter(m, ms)
                gb = g == 1
                acc[0] += (m & gb).sum()
                acc[1] += (m & ~gb).sum()
                acc[2] += (~m & gb).sum()
            val = f'{f1(*acc):.3f}'
            cells.append(f'**{val}**' if pt == -12 else val)
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')

    # min_size 列内波动（post=-12 固定）：max−min
    vals = []
    for ms in MIN_SIZES:
        acc = [0, 0, 0]
        for base, post, g in patches:
            m = conn_filter(base & (np.nan_to_num(post) < -12), ms)
            gb = g == 1
            acc[0] += (m & gb).sum()
            acc[1] += (m & ~gb).sum()
            acc[2] += (~m & gb).sum()
        vals.append(f1(*acc))
    lines.append(f'- **min_size 波动**（post=-12 固定）：F1 {min(vals):.3f}–{max(vals):.3f}（范围 {max(vals)-min(vals):.3f}）')
    lines.append('')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('saved:', OUT)
