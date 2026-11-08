#!/usr/bin/env python3
"""Phase 1d · 逐事件城区表现 —— 跨检测器家族稳健性检验。

问题：Phase 1c 发现"分支能力强烈依赖事件"（16 事件 F1 跨度 0.006–0.619）。
但那只用了一个家族（coh_union, P=0.6, D=0.30, pm=-3）。需要检验：
**这个事件依赖形态是数据的性质，还是该组设置的偶然？**

做法：对 4 个家族（极化 × 亮度约束）用同一 P/D，逐事件累计混淆矩阵，
比较各家族的事件排序是否一致（Spearman 秩相关）。

产出：docs/03_结果/flood_ge_urban_by_event_xfam.md
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
                   'flood_ge_urban_by_event_xfam.md')

P_FIX, D_FIX = 0.6, 0.30
FAMILIES = {
    'union_pm3': ('union', -3.0),
    'union_pm6': ('union', -6.0),
    'vh_pm6':    ('vh',    -6.0),
    'vv_pm3':    ('vv',    -3.0),
}


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0


def event_of(path):
    m = re.search(r'(\d{8}_[A-Za-z0-9_]+?)_ID_', os.path.basename(path))
    return m.group(1) if m else 'UNKNOWN'


def spearman(a, b):
    """秩相关（无 scipy 依赖）。"""
    def rank(v):
        order = np.argsort(np.argsort(v))
        return order.astype(float)
    ra, rb = rank(np.asarray(a)), rank(np.asarray(b))
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def main():
    samples = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')
    print(f'[xfam] 样本总数: {len(samples)}', flush=True)

    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
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
                post = fin(sar[7])
                m_vv = ((pvv - cvv) > D_FIX) & (pvv > P_FIX)
                m_vh = ((pvh - cvh) > D_FIX) & (pvh > P_FIX)
                for fam, (pol, pm) in FAMILIES.items():
                    base = m_vv if pol == 'vv' else m_vh if pol == 'vh' else (m_vv | m_vh)
                    m = base & (post > pm)
                    cm = acc[ev][fam]
                    cm[0] += int((m & urban).sum())
                    cm[1] += int((m & ~urban).sum())
                    cm[2] += int((~m & urban).sum())
        if (i + 1) % 1500 == 0:
            print(f'  [xfam] {i+1}/{len(samples)}', flush=True)

    events = sorted(acc.keys(), key=lambda e: -f1(*acc[e]['union_pm3']))
    fams = list(FAMILIES.keys())

    # 秩相关矩阵
    vecs = {f: [f1(*acc[e][f]) for e in events] for f in fams}
    rho = {}
    for a in fams:
        for b in fams:
            rho[(a, b)] = spearman(vecs[a], vecs[b])

    L = ['# 逐事件城区表现 · 跨家族稳健性', '',
         '> 分支单独谓词；官方 train+valid；P=0.6, D=0.30；每家族各自的事件排序。',
         '> 若各家族的事件排序高度一致（Spearman ρ 高），则"事件依赖"是**数据的性质**，',
         '> 而非某一组设置的偶然。', '',
         '## 逐事件 F1（按 union_pm3 降序）', '',
         '| 事件 | 城区GT像素 | ' + ' | '.join(fams) + ' |',
         '|:--|--:|' + '--:|' * len(fams)]
    for e in events:
        L.append(f'| {e} | {npx[e]} | ' +
                 ' | '.join(f'{f1(*acc[e][f]):.3f}' for f in fams) + ' |')
    L.append('')
    L += ['## 各家族事件排序的一致性（Spearman ρ）', '',
          '| | ' + ' | '.join(fams) + ' |', '|:--|' + '--:|' * len(fams)]
    for a in fams:
        L.append(f'| {a} | ' + ' | '.join(f'{rho[(a,b)]:.3f}' for b in fams) + ' |')
    L.append('')

    off = [rho[(a, b)] for a in fams for b in fams if a != b]
    L += ['## 判定', '',
          f'- 非对角 Spearman ρ：最低 {min(off):.3f} / 中位 {np.median(off):.3f} / 最高 {max(off):.3f}',
          f'- 各族 F1 跨事件极差：' + '；'.join(
              f'{f} {min(vecs[f]):.3f}–{max(vecs[f]):.3f}' for f in fams),
          '',
          '> 判据：ρ 中位 ≥0.7 → 事件依赖是数据性质，稳健；',
          '> ρ 低 → 事件排序随家族变化，Phase 1c 的结论只对该设置成立。',
          '']

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('saved:', OUT, flush=True)
    print(f'非对角 rho: min={min(off):.3f} median={np.median(off):.3f} max={max(off):.3f}',
          flush=True)


if __name__ == '__main__':
    main()
