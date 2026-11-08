#!/usr/bin/env python3
"""Phase 1c · 城区分支按**事件**拆分的校准表现（分支单独谓词）。

动机：测试集上分支单独表现两极——Jubba 精度 0.57–0.62、Weihui 0.048。
若校准集 0.219 是"好事件"与"坏事件"的混合均值，则这个均值掩盖了
场景依赖的真实结构。

本脚本把官方 train+valid 按文件名前缀（YYYYMMDD_EventName）分组，
用**分支单独**谓词（与测试集分支单独口径一致）逐事件累计混淆矩阵。

检测器固定：P=0.6, D=0.30, post_vv > -3 dB（coh_union），与旧校准最优点相同。

产出：docs/03_结果/flood_ge_urban_by_event.md
"""
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rasterio
from data_loader import parse_split
from flood_strategies import CO_PRE_VV, CO_CO_VV, CO_PRE_VH, CO_CO_VH

_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.abspath(os.path.join(_REPRO_DIR, os.pardir, 'urban_sar_floods'))
ROOT = os.environ.get('USF_ROOT', _DATA_ROOT)
OUT = os.path.join(_REPRO_DIR, os.pardir, 'docs', '03_结果',
                   'flood_ge_urban_by_event.md')

P_FIX, D_FIX, PM_FIX = 0.6, 0.30, -3.0


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0


def event_of(path):
    m = re.search(r'(\d{8}_[A-Za-z0-9_]+?)_ID_', os.path.basename(path))
    return m.group(1) if m else 'UNKNOWN'


def main():
    samples = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')
    print(f'[by-event] 样本总数: {len(samples)}', flush=True)

    acc = defaultdict(lambda: [0, 0, 0])   # event -> [tp, fp, fn] (urban, branch-only)
    nquad = defaultdict(int)
    nchip = defaultdict(int)
    npx = defaultdict(int)
    nall = defaultdict(int)

    for i, (sar_p, gt_p) in enumerate(samples):
        ev = event_of(sar_p)
        nchip[ev] += 1
        with rasterio.open(gt_p) as ds:
            g = ds.read(1)
        H, W = g.shape
        for y0 in range(0, H - 255, 256):
            for x0 in range(0, W - 255, 256):
                q = g[y0:y0 + 256, x0:x0 + 256]
                if (q > 0).sum() < 100:
                    continue
                with rasterio.open(sar_p) as ds:
                    sar = ds.read(window=((y0, y0 + 256), (x0, x0 + 256))).astype(np.float32)
                urban = (q == 2)
                fin = lambda a: np.nan_to_num(a, nan=np.nanmedian(a))
                pvv, cvv = fin(sar[CO_PRE_VV]), fin(sar[CO_CO_VV])
                pvh, cvh = fin(sar[CO_PRE_VH]), fin(sar[CO_CO_VH])
                m = (((pvv - cvv) > D_FIX) & (pvv > P_FIX)) | \
                    (((pvh - cvh) > D_FIX) & (pvh > P_FIX))
                m &= (fin(sar[7]) > PM_FIX)
                cm = acc[ev]
                cm[0] += int((m & urban).sum())
                cm[1] += int((m & ~urban).sum())
                cm[2] += int((~m & urban).sum())
                nquad[ev] += 1
                npx[ev] += int(urban.sum())
                nall[ev] += int(urban.size)
        if (i + 1) % 1000 == 0:
            print(f'  [by-event] {i+1}/{len(samples)}', flush=True)

    rows = []
    for ev, (tp, fp, fn) in acc.items():
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        rows.append((f1(tp, fp, fn), p_, r_, ev, tp, fp, fn))
    rows.sort(reverse=True, key=lambda x: x[0])

    tot = [sum(acc[e][k] for e in acc) for k in range(3)]
    tot_f1 = f1(*tot)

    L = ['# 城区分支按事件拆分（分支单独谓词 · 官方 train+valid）', '',
         '> 协议：512 chip → 4×256 象限，(g>0)>=100，像素聚合，目标 g==2。',
         '> **分支单独谓词**（不含 config F），与测试集分支单独口径一致。',
         f'> 检测器固定：P={P_FIX}, D={D_FIX}, post_vv > {PM_FIX} dB（coh_union）。',
         '> 事件由文件名前缀 (YYYYMMDD_EventName) 识别。',
         '',
         f'## 汇总（全部 {len(acc)} 个事件合并）', '',
         f'- 城区 F1 **{tot_f1:.3f}**（精度 {tot[0]/(tot[0]+tot[1]):.3f} / '
         f'召回 {tot[0]/(tot[0]+tot[2]):.3f}）',
         f'- TP {tot[0]} / FP {tot[1]} / FN {tot[2]}',
         '',
         '## 逐事件（按 F1 降序）', '',
         '| 事件 | 城区F1 | 精度 | 召回 | TP | FP | FN | 城区GT像素 | 非城区虚警密度(‰) | 象限数 |',
         '|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for f1v, p_, r_, ev, tp, fp, fn in rows:
        non_urb = max(nall[ev] - npx[ev], 1)
        L.append(f'| {ev} | {f1v:.3f} | {p_:.3f} | {r_:.3f} | {tp} | {fp} | {fn} | '
                 f'{npx[ev]} | {1000.0*fp/non_urb:.1f} | {nquad[ev]} |')
    L.append('')

    # 分布形态
    fs = [r[0] for r in rows]
    L += ['## 分布形态', '',
          f'- 最高 {max(fs):.3f} / 最低 {min(fs):.3f} / 中位 {np.median(fs):.3f}',
          f'- ≥0.4 的事件数: {sum(1 for v in fs if v >= 0.4)}',
          f'- <0.1 的事件数: {sum(1 for v in fs if v < 0.1)}',
          '',
          '> 判据：若分布明显双峰（部分事件 ≥0.4、部分 <0.1）→ 分支表现是**场景依赖**，',
          '> 校准集的合并均值掩盖了真实结构；若集中在 0.2 附近 → 无场景结构。',
          '']

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('saved:', OUT, flush=True)
    print(f'ALL F1={tot_f1:.3f}  top={rows[0][3]}({rows[0][0]:.3f})  '
          f'bottom={rows[-1][3]}({rows[-1][0]:.3f})', flush=True)


if __name__ == '__main__':
    main()
