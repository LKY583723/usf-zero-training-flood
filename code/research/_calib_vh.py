"""VV×VH 双极化暗约束校准：post_vv<VV_T & post_vh<VH_T，conn=500。

基线对比：post_vv<-12 单独 = 0.676（open）。目标：AND 双极化后能否 >0.676。
"""
import os
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import rasterio
from data_loader import parse_split
from flood_strategies import (strategy_int, strategy_chg, combine_pol,
                              conn_filter, INT_POST_VV, INT_POST_VH)

ROOT = 'D:/DisasterM3/UrbanSARFloods/urban_sar_floods'
OUT = 'D:/DisasterM3/experiment1/results/flood_ge_improve_calib.md'

VV_THR = [-10, -11, -12, -13, -14]
VH_THR = [-14, -16, -18, -20, -22]
CONN = 500


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


def main():
    samples = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')
    acc = {(a, b): {'open': [0, 0, 0], 'all': [0, 0, 0]} for a in VV_THR for b in VH_THR}
    # 单极化参考
    ref_vv = {a: [0, 0, 0] for a in VV_THR}
    ref_vh = {b: [0, 0, 0] for b in VH_THR}
    n_quad = 0
    for i, (sar_p, gt_p) in enumerate(samples):
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
                base = combine_pol(sar, lambda s, p: strategy_int(s, p), 'or') & \
                    combine_pol(sar, lambda s, p: strategy_chg(s, p), 'or')
                vv = np.nan_to_num(sar[INT_POST_VV]) < np.array(VV_THR)[:, None, None]
                vh = np.nan_to_num(sar[INT_POST_VH]) < np.array(VH_THR)[:, None, None]
                gb = q == 1
                ga = q > 0
                n_quad += 1
                for ai, a in enumerate(VV_THR):
                    for bi, b in enumerate(VH_THR):
                        m = conn_filter(base & vv[ai] & vh[bi], CONN)
                        c = acc[(a, b)]
                        c['open'][0] += (m & gb).sum(); c['open'][1] += (m & ~gb).sum(); c['open'][2] += (~m & gb).sum()
                        c['all'][0] += (m & ga).sum(); c['all'][1] += (m & ~ga).sum(); c['all'][2] += (~m & ga).sum()
                # 单极化参考（VV only / VH only）
                for ai, a in enumerate(VV_THR):
                    m = conn_filter(base & vv[ai], CONN)
                    r = ref_vv[a]
                    r[0] += (m & gb).sum(); r[1] += (m & ~gb).sum(); r[2] += (~m & gb).sum()
                for bi, b in enumerate(VH_THR):
                    m = conn_filter(base & vh[bi], CONN)
                    r = ref_vh[b]
                    r[0] += (m & gb).sum(); r[1] += (m & ~gb).sum(); r[2] += (~m & gb).sum()
        if (i + 1) % 300 == 0:
            print(f'  [calib] {i+1}/{len(samples)} 样本, {n_quad} 象限', flush=True)

    lines = [f'# VV×VH 双极化暗约束校准（train+valid，{n_quad} 象限，conn={CONN}）', '',
             '## 双极化 open F1（行=VV 阈值，列=VH 阈值）', '',
             '| VV\\VH | ' + ' | '.join(str(b) for b in VH_THR) + ' |', '|:--|' + '|:--:|' * len(VH_THR)]
    best = []
    for a in VV_THR:
        cells = [str(a)]
        for b in VH_THR:
            v = f1(*acc[(a, b)]['open'])
            cells.append(f'{v:.3f}')
            best.append((v, a, b))
        lines.append('| ' + ' | '.join(cells) + ' |')
    best.sort(reverse=True)
    lines += ['', '## 单极化参考（open F1）', '']
    lines.append('| VV 单独 | VH 单独 |')
    lines.append('|:--|:--|')
    for a in VV_THR:
        lines.append(f'| {a}: {f1(*ref_vv[a]):.3f} | — |')
    for b in VH_THR:
        lines.append(f'| — | {b}: {f1(*ref_vh[b]):.3f} |')
    lines += ['', '## 最优组合', '']
    for v, a, b in best[:8]:
        lines.append(f'- (VV={a}, VH={b}): open {v:.3f}')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved:', OUT, flush=True)
    for v, a, b in best[:8]:
        print(f'  (VV={a}, VH={b}): open {v:.3f}', flush=True)


if __name__ == '__main__':
    main()
