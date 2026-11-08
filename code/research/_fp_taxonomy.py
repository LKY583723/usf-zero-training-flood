#!/usr/bin/env python3
"""Phase 3 · 验证门分层 FP 消解表（机制量化）。

对 4 测试事件（open 目标 GT==1）逐层累计混淆矩阵：
  A    = S_int（单策略，消融 A）
  cand = S_int ∩ S_chg
  post = cand ∩ (post_VV < −12 dB)
  F    = conn_filter(post, 500 px)   ← config F

目的：展示"验证门=假阳率相乘、召回几乎不降"的逐层量化，供论文机制/讨论节使用。

协议：testing_case_256、(g>0)>=100、像素聚合、Jubba 混淆矩阵合并（同 _eval_3event.py）。
产出：docs/03_结果/flood_ge_fp_taxonomy.md
"""
import glob
import os
import re
import sys

_REPRO = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_REPRO)
sys.path.insert(0, _REPRO)

import numpy as np
import rasterio
from flood_strategies import all_strategies, conn_filter

DATA = os.environ.get('USF_ROOT', _DATA)
PATCHES = os.path.join(DATA, 'testing_case_256')
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
CONN_MIN = 500
POST_THR = -12.0
STAGES = ['A', 'cand', 'post', 'F']


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def stage_masks(sar):
    s = all_strategies(sar)
    a = s['int']
    cand = s['int'] & s['chg']
    post = cand & (np.nan_to_num(sar[7]) < POST_THR)
    f = conn_filter(post, CONN_MIN)
    return {'A': a, 'cand': cand, 'post': post, 'F': f}


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


def main():
    acc = {}
    for sc in SCENES:
        patch_dir = os.path.join(PATCHES, sc)
        files = sorted(glob.glob(os.path.join(patch_dir, '*_SAR.tif')))
        gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
        if not (files and os.path.exists(gt_path)):
            continue
        with rasterio.open(gt_path) as ds:
            gt_full = ds.read(1)
        acc[sc] = {st: [0, 0, 0] for st in STAGES}
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
            for st, pred in stage_masks(sar).items():
                cm = acc[sc][st]
                cm[0] += int((pred & (g == 1)).sum())
                cm[1] += int((pred & (g != 1)).sum())
                cm[2] += int((~pred & (g == 1)).sum())

    lines = ['# 验证门分层 FP 消解表（open 目标 GT==1 · 测试事件）', '']
    lines.append('> 协议：testing_case_256，(g>0)>=100，像素聚合。')
    lines.append('> 层级：A = S_int（单策略）→ cand = S_int∩S_chg → post = +post_VV<−12dB → F = +连通域≥500px（= config F）。')
    lines.append('> "本层砍FP" = 上一层 FP − 本层 FP；"累计砍FP%" 相对 A 层。')
    lines.append('')
    for sc in SCENES:
        lines.append(f'## {sc}')
        lines.append('')
        lines.append('| 层 | TP | FP | FN | openF1 | 精度 | 召回 | 本层砍FP | 累计砍FP% |')
        lines.append('|:--|--:|--:|--:|:--:|:--:|:--:|--:|--:|')
        prev_fp = None
        base_fp = acc[sc]['A'][1]
        for st in STAGES:
            tp, fp, fn = acc[sc][st]
            cut = (prev_fp - fp) if prev_fp is not None else 0
            cum = (1 - fp / base_fp) * 100 if base_fp else 0
            p_ = tp / (tp + fp) if tp + fp else 0
            r_ = tp / (tp + fn) if tp + fn else 0
            lines.append(f'| {st} | {tp} | {fp} | {fn} | {f1(tp, fp, fn):.3f} | {p_:.3f} | {r_:.3f} | {cut} | {cum:.1f}% |')
            prev_fp = fp
        lines.append('')
    lines.append('## 3 事件均值（Jubba 合并）')
    lines.append('')
    jubba = {st: [0, 0, 0] for st in STAGES}
    for st in STAGES:
        for i in range(3):
            jubba[st][i] = (acc['20231201_Jubba_1'][st][i]
                            + acc['20231201_Jubba_2'][st][i])
    events = {'Weihui': acc['20210727_Weihui'], 'NovaKakhovka': acc['20230609_NovaKakhovka'],
              'Jubba': jubba}
    lines.append('| 层 | 平均 openF1 | 平均精度 | 平均召回 | 累计砍FP%(Jubba) |')
    lines.append('|:--|:--:|:--:|:--:|:--:|')
    for st in STAGES:
        f1s = [f1(*events[k][st]) for k in events]
        p_s = []
        for k in events:
            tp, fp, fn = events[k][st]
            p_s.append(tp / (tp + fp) if tp + fp else 0)
        r_s = []
        for k in events:
            tp, fp, fn = events[k][st]
            r_s.append(tp / (tp + fn) if tp + fn else 0)
        cum = (1 - jubba[st][1] / jubba['A'][1]) * 100 if jubba['A'][1] else 0
        lines.append(f'| {st} | {np.mean(f1s):.3f} | {np.mean(p_s):.3f} | {np.mean(r_s):.3f} | {cum:.1f}% |')
    lines.append('')

    out = 'D:/DisasterM3/UrbanSARFloods/docs/03_结果/flood_ge_fp_taxonomy.md'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved:', out, flush=True)


if __name__ == '__main__':
    main()
