"""GE-F 改进变体校准（官方 train+valid，与 _calib_threshold.py 同协议）。

基线：int∩chg + post_vv<-12 + conn>=500（open F1 ≈ 0.676）
新变体（全部零训练、参数在校准集上选）：
  VH1..VH4   加 post_vh<THR 暗约束（VH 暗水）
  REL3..REL7 相对暗度 post_vv < 局部中位数 - delta（每 patch 自动）
  MOR3..MOR7 形态学闭合先连接细洪水线，再连通域过滤
  BEST       各最优分量的组合
输出：open F1 / all F1 对比表。
"""
import os
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import rasterio
import cv2
from data_loader import parse_split
from flood_strategies import (strategy_int, strategy_chg, combine_pol,
                              conn_filter, INT_POST_VV, INT_POST_VH)

ROOT = 'D:/DisasterM3/UrbanSARFloods/urban_sar_floods'
OUT = 'D:/DisasterM3/experiment1/results/flood_ge_improve_calib.md'


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


def closing(mask, k):
    kern = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kern).astype(bool)


def main():
    samples = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')

    # 变体定义：name -> 构造函数(base, post_vv, post_vh, med)
    def mk_vh(thr):
        return lambda b, vv, vh, med: b & (vh < thr)

    def mk_rel(delta):
        return lambda b, vv, vh, med: b & (vv < med - delta)

    def mk_morph(k):
        return lambda b, vv, vh, med: closing(b, k)

    VARIANTS = [('baseline', lambda b, vv, vh, med: b)]
    for thr in (-16, -18, -20, -22):
        VARIANTS.append((f'VH{int(-thr)}', mk_vh(thr)))
    for delta in (3, 4, 5, 6, 7):
        VARIANTS.append((f'REL{delta}', mk_rel(delta)))
    for k in (3, 5, 7):
        VARIANTS.append((f'MOR{k}', mk_morph(k)))

    acc = {name: {'open': [0, 0, 0], 'all': [0, 0, 0]} for name, _ in VARIANTS}
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
                s_int = combine_pol(sar, lambda s, p: strategy_int(s, p), 'or')
                s_chg = combine_pol(sar, lambda s, p: strategy_chg(s, p), 'or')
                base = s_int & s_chg
                vv = np.nan_to_num(sar[INT_POST_VV])
                vh = np.nan_to_num(sar[INT_POST_VH])
                med = np.nanmedian(sar[INT_POST_VV])
                n_quad += 1
                for name, fn in VARIANTS:
                    m = conn_filter(fn(base, vv, vh, med), 500)
                    a = acc[name]
                    a['open'][0] += (m & (q == 1)).sum()
                    a['open'][1] += (m & (q != 1)).sum()
                    a['open'][2] += (~m & (q == 1)).sum()
                    a['all'][0] += (m & (q > 0)).sum()
                    a['all'][1] += (m & (q == 0)).sum()
                    a['all'][2] += (~m & (q > 0)).sum()
        if (i + 1) % 300 == 0:
            print(f'  [calib] {i+1}/{len(samples)} 样本, {n_quad} 洪水象限', flush=True)

    print(f'[calib] 洪水象限: {n_quad}', flush=True)
    lines = [f'# GE-F 改进变体校准（官方 train+valid，{n_quad} 象限）', '',
             '> 协议同 _calib_threshold.py：256² 象限像素聚合 F1，conn=500。', '',
             '| 变体 | 规则 | open F1 | all F1 |', '|:--|:--|:--:|:--:|']
    rows = []
    for name, fn in VARIANTS:
        a = acc[name]
        lines.append(f'| {name} | {fn.__name__} | {f1(*a["open"]):.3f} | {f1(*a["all"]):.3f} |')
        rows.append((name, f1(*a['open']), f1(*a['all'])))
    lines.insert(0, '')  # placeholder
    # 排序表（按 open F1 降序）
    lines = lines[:3] + ['| 排名 | 变体 | open F1 | all F1 |', '|:--|:--|:--:|:--:|']
    for rank, (name, of, af) in enumerate(sorted(rows, key=lambda r: -r[1]), 1):
        lines.append(f'| {rank} | {name} | {of:.3f} | {af:.3f} |')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved:', OUT, flush=True)
    for rank, (name, of, af) in enumerate(sorted(rows, key=lambda r: -r[1]), 1):
        print(f'  {rank}. {name}: open {of:.3f} all {af:.3f}', flush=True)


if __name__ == '__main__':
    main()
