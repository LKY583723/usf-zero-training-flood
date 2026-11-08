#!/usr/bin/env python3
"""诊断 · 逐事件城区表现的机制分解。

Phase 1c/1d 已确立"分支能力强烈依赖事件"（跨家族稳健）。
但**为什么**未知：此前的"干旱 vs 农业"假说已被虚警密度否掉。

本脚本把检测器的三个条件拆开，分别统计它们在**城区 GT 像素**上的满足率：
    C1: pre_coh > 0.6      （灾前高相干 = 建筑稳定散射体）
    C2: Δγ > 0.3           （跨灾期显著去相干）
    C3: post_vv > -3 dB    （灾后亮回波 = 二面角）
recall = frac(C1 ∧ C2 ∧ C3)，因此可定位**哪个条件是瓶颈**。

同时对**非城区像素**统计同一三条件满足率 —— 即假阳的原料密度。

产出：docs/03_结果/flood_ge_urban_mechanism.md
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
                   'flood_ge_urban_mechanism.md')

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
    print(f'[mech] 样本总数: {len(samples)}', flush=True)

    # 城区 GT 像素上：各条件的满足计数
    u = defaultdict(lambda: {'n': 0, 'c1': 0, 'c2': 0, 'c3': 0, 'all': 0})
    # 非城区像素：三条件全满足的计数（假阳原料）
    b = defaultdict(lambda: {'n': 0, 'all': 0})

    for i, (sar_p, gt_p) in enumerate(samples):
        ev = event_of(sar_p)
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
                post = fin(sar[7])
                # 与 coh_union 家族一致：任一极化满足即可
                c1 = (pvv > P_FIX) | (pvh > P_FIX)
                c2 = ((pvv - cvv) > D_FIX) | ((pvh - cvh) > D_FIX)
                c3 = post > PM_FIX
                allc = c1 & c2 & c3
                a = u[ev]
                a['n'] += int(urban.sum())
                a['c1'] += int((c1 & urban).sum())
                a['c2'] += int((c2 & urban).sum())
                a['c3'] += int((c3 & urban).sum())
                a['all'] += int((allc & urban).sum())
                nb = ~urban
                b[ev]['n'] += int(nb.sum())
                b[ev]['all'] += int((allc & nb).sum())
        if (i + 1) % 1500 == 0:
            print(f'  [mech] {i+1}/{len(samples)}', flush=True)

    rows = []
    for ev in u:
        a, bb = u[ev], b[ev]
        n = max(a['n'], 1)
        rec = a['all'] / n
        fprate = bb['all'] / max(bb['n'], 1)
        prec_est = a['all'] / max(a['all'] + bb['all'], 1)
        rows.append((rec, ev, a['c1'] / n, a['c2'] / n, a['c3'] / n, n,
                     rec, fprate, prec_est, f1(a['all'], bb['all'], n - a['all'])))
    rows.sort(key=lambda r: -r[5])
    rows.sort(key=lambda r: -r[9])

    L = ['# 城区分支 · 逐事件机制分解（官方 train+valid）', '',
         '> 分支单独谓词。检测器三条件（与 coh_union 家族一致，任一极化满足即可）：',
         f'> C1 = pre_coh > {P_FIX}；C2 = Δγ > {D_FIX}；C3 = post_vv > {PM_FIX} dB。',
         '> 各列均为**城区 GT 像素上的满足率**；"三条件全满足率"即召回。',
         '> 末列为非城区像素上的同一全满足率（假阳原料密度）。',
         '',
         '| 事件 | 城区GT像素 | C1满足 | C2满足 | C3满足 | **三条件全满足(=召回)** | 非城区全满足率 | 估算F1 |',
         '|:--|--:|--:|--:|--:|--:|--:|--:|']
    for rec, ev, c1, c2, c3, n, r_, fpr, prec, f1v in rows:
        L.append(f'| {ev} | {n} | {c1:.3f} | {c2:.3f} | {c3:.3f} | **{r_:.3f}** | '
                 f'{fpr:.4f} | {f1v:.3f} |')
    L.append('')

    # 瓶颈识别
    L += ['## 瓶颈诊断（每个事件最卡的条件）', '',
          '| 事件 | 最低满足率条件 | 该条件满足率 |', '|:--|:--|--:|']
    for rec, ev, c1, c2, c3, n, r_, fpr, prec, f1v in rows:
        conds = [('C1 pre_coh', c1), ('C2 Δγ', c2), ('C3 post', c3)]
        lo = min(conds, key=lambda x: x[1])
        L.append(f'| {ev} | {lo[0]} | {lo[1]:.3f} |')
    L.append('')

    L += ['## 判定', '',
          '> 若不同事件的瓶颈条件不同（有的卡 C1、有的卡 C2），说明"事件依赖"来自',
          '> **物理前提本身在不同场景成立程度不同**，而非单一阈值问题。',
          '']

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('saved:', OUT, flush=True)
    for rec, ev, c1, c2, c3, n, r_, fpr, prec, f1v in rows[:3] + rows[-3:]:
        print(f'  {ev:24s} C1={c1:.2f} C2={c2:.2f} C3={c3:.2f} R={r_:.3f} '
              f'bgrate={fpr:.4f}', flush=True)


if __name__ == '__main__':
    main()
