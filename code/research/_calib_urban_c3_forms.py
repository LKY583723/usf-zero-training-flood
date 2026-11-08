#!/usr/bin/env python3
"""Phase 1e · C3（亮回波约束）的形式检验。

机制分解（flood_ge_urban_mechanism.md）显示：
  C2（Δγ>0.3）在城区 GT 上 16 事件全部 =1.00 —— 相干损失是普适标记，从不是瓶颈；
  C3（post_vv > -3 dB）是 16 事件中 15 个的瓶颈，满足率仅 0.132–0.705。

假设：C3 用**绝对亮度**表述是错的。"城区洪水产生亮二面角"这个物理命题说的是
**淹没前后的变化**，不是绝对电平。故改用相对变化 ΔI = post_vv − pre_vv。

本脚本在固定 C1、C2 的前提下，比较 C3 的多种形式，逐事件 + 合并报告。

产出：docs/03_结果/flood_ge_urban_c3_forms.md
"""
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rasterio
from data_loader import parse_split
from flood_strategies import (CO_PRE_VV, CO_CO_VV, CO_PRE_VH, CO_CO_VH,
                              INT_PRE_VV, INT_POST_VV)

_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.abspath(os.path.join(_REPRO_DIR, os.pardir, 'urban_sar_floods'))
ROOT = os.environ.get('USF_ROOT', _DATA_ROOT)
OUT = os.path.join(_REPRO_DIR, os.pardir, 'docs', '03_结果',
                   'flood_ge_urban_c3_forms.md')

P_FIX, D_FIX = 0.6, 0.30
# C3 形式：('名称', 函数(post, pre) -> bool)
C3 = [
    ('post>-3  (当前)',      lambda po, pr: po > -3.0),
    ('post>-6',              lambda po, pr: po > -6.0),
    ('post-pre>0',           lambda po, pr: (po - pr) > 0.0),
    ('post-pre>1',           lambda po, pr: (po - pr) > 1.0),
    ('post-pre>2',           lambda po, pr: (po - pr) > 2.0),
    ('无 C3',                lambda po, pr: np.ones_like(po, dtype=bool)),
]


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0


def event_of(path):
    m = re.search(r'(\d{8}_[A-Za-z0-9_]+?)_ID_', os.path.basename(path))
    return m.group(1) if m else 'UNKNOWN'


def main():
    samples = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')
    print(f'[c3] 样本总数: {len(samples)}', flush=True)

    names = [c[0] for c in C3]
    acc = defaultdict(lambda: {n: [0, 0, 0] for n in names})
    npx = defaultdict(int)

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
                npx[ev] += int(urban.sum())
                fin = lambda a: np.nan_to_num(a, nan=np.nanmedian(a))
                pvv, cvv = fin(sar[CO_PRE_VV]), fin(sar[CO_CO_VV])
                pvh, cvh = fin(sar[CO_PRE_VH]), fin(sar[CO_CO_VH])
                po = fin(sar[INT_POST_VV])
                pr = fin(sar[INT_PRE_VV])
                core = ((((pvv - cvv) > D_FIX) & (pvv > P_FIX)) |
                        (((pvh - cvh) > D_FIX) & (pvh > P_FIX)))
                for nm, fn in C3:
                    m = core & fn(po, pr)
                    cm = acc[ev][nm]
                    cm[0] += int((m & urban).sum())
                    cm[1] += int((m & ~urban).sum())
                    cm[2] += int((~m & urban).sum())
        if (i + 1) % 1500 == 0:
            print(f'  [c3] {i+1}/{len(samples)}', flush=True)

    events = sorted(acc.keys(), key=lambda e: -f1(*acc[e][names[0]]))

    L = ['# C3（亮回波约束）形式检验 —— 固定 C1、C2', '',
         f'> C1 = pre_coh > {P_FIX}（任一极化）；C2 = Δγ > {D_FIX}（任一极化）。',
         '> 分支单独谓词；官方 train+valid；每格为城区 F1。',
         '> **假设**：机制分解显示 C3 是主要瓶颈（满足率 0.132–0.705），',
         '> 而"城区洪水产生亮二面角"说的是**变化**而非绝对电平，故试相对形式。',
         '',
         '## 逐事件城区 F1', '',
         '| 事件 | 城区GT | ' + ' | '.join(names) + ' |',
         '|:--|--:|' + '--:|' * len(names)]
    for e in events:
        L.append(f'| {e} | {npx[e]} | ' +
                 ' | '.join(f'{f1(*acc[e][n]):.3f}' for n in names) + ' |')
    L.append('')

    # 合并
    L += ['## 合并（全事件像素聚合）', '',
          '| C3 形式 | 城区F1 | 精度 | 召回 |', '|:--|--:|--:|--:|']
    pooled = {}
    for n in names:
        t = [sum(acc[e][n][k] for e in acc) for k in range(3)]
        pooled[n] = t
        L.append(f'| {n} | {f1(*t):.3f} | '
                 f'{t[0]/(t[0]+t[1]) if t[0]+t[1] else 0:.3f} | '
                 f'{t[0]/(t[0]+t[2]) if t[0]+t[2] else 0:.3f} |')
    L.append('')

    # 大样本事件子集（城区GT>=5000）
    big = [e for e in events if npx[e] >= 5000]
    L += [f'## 仅大样本事件（城区GT ≥5000，共 {len(big)} 个）', '',
          '| C3 形式 | 城区F1 | 精度 | 召回 |', '|:--|--:|--:|--:|']
    for n in names:
        t = [sum(acc[e][n][k] for e in big) for k in range(3)]
        L.append(f'| {n} | {f1(*t):.3f} | '
                 f'{t[0]/(t[0]+t[1]) if t[0]+t[1] else 0:.3f} | '
                 f'{t[0]/(t[0]+t[2]) if t[0]+t[2] else 0:.3f} |')
    L.append('')

    base = 'post>-3  (当前)'
    L += ['## 判定', '',
          f'- 当前形式（{base}）合并 F1 {f1(*pooled[base]):.3f}（大样本子集 '
          f'{f1(*[sum(acc[e][base][k] for e in big) for k in range(3)]):.3f}）']
    best = max(names, key=lambda n: f1(*pooled[n]))
    L += [f'- 合并最优形式: **{best}** → F1 {f1(*pooled[best]):.3f}',
          f'- 相对变化是否优于绝对电平：见上表。',
          '> 注意：若某形式只在大样本子集上占优，更可信（小样本事件噪声大）。',
          '']

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('saved:', OUT, flush=True)
    for n in names:
        t = pooled[n]
        tb = [sum(acc[e][n][k] for e in big) for k in range(3)]
        print(f'  {n:20s} pooled F1={f1(*t):.3f}  big-only F1={f1(*tb):.3f}', flush=True)


if __name__ == '__main__':
    main()
