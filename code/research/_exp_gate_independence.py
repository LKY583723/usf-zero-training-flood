#!/usr/bin/env python3
"""实验 A · 验证门各约束的假阳集合独立性量化。

论文 §II-B 现在有一句**未经检验**的断言：
  "Because these signals come from different physical measurements,
   their conjunction multiplies false-positive rates."

本脚本把它变成可测命题。对 config F 的三条约束（S_chg、post<-12dB、连通域≥500px），
在**同一候选池**（S_int ∩ 非洪水）上分别测：

  边际存活率 m_k = |S0 ∩ C_k| / |S0|
  实测联合存活率 J_obs = |S0 ∩ C1 ∩ C2 ∩ C3| / |S0|
  独立预测       J_ind = m1 · m2 · m3

判读：
  J_obs ≈ J_ind  → 假阳集合**独立**，"合取使假阳相乘"成立
  J_obs > J_ind  → 正相关，约束**冗余**（杀的假阳重叠），该断言过强
  J_obs < J_ind  → **次独立**，各约束杀掉互不相交的假阳子集，复合优于独立假设

口径：testing_case_256、(g>0)>=100、像素聚合；非洪水 = (GT != 1)，与 open 评测一致。

产出：docs/03_结果/flood_ge_gate_independence.md
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
POST_THR = -12.0
CONN_MIN = 500

OUT = os.path.join(_REPRO, os.pardir, 'docs', '03_结果',
                   'flood_ge_gate_independence.md')


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def main():
    # acc[scene] = dict(s0, s1, s2, s3, m1n, m2n, m3n, rm1, rm2, rm3, inter)
    acc = {}

    for sc in SCENES:
        files = sorted(glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')))
        gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
        if not (files and os.path.exists(gt_path)):
            print(f'[skip] {sc}', flush=True)
            continue
        with rasterio.open(gt_path) as ds:
            gt_full = ds.read(1)

        d = dict(s0=0, s1=0, s2=0, s3=0, n1=0, n2=0, n3=0)
        # 两两交叠（被各约束剔除的假阳集合之间的 Jaccard）
        pair = {(1, 2): [0, 0], (1, 3): [0, 0], (2, 3): [0, 0]}

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
            nonflood = (g != 1)
            S0 = s['int'] & nonflood          # 候选池：单策略的假阳
            if not S0.any():
                continue
            post = np.nan_to_num(sar[7], nan=np.nanmedian(sar[7]))
            C1 = s['chg']
            C2 = (post < POST_THR)
            C3 = conn_filter(S0, CONN_MIN)    # 连通域作为独立滤波器，作用于同一池
            S1, S2, S3 = S0 & C1, S0 & C2, S0 & C3
            Sall = S1 & S2 & S3

            d['s0'] += int(S0.sum())
            d['s1'] += int(S1.sum())
            d['s2'] += int(S2.sum())
            d['s3'] += int(S3.sum())
            d['n1'] += int((S0 & ~C1).sum())
            d['n2'] += int((S0 & ~C2).sum())
            d['n3'] += int((S0 & ~C3).sum())

            R = {1: S0 & ~C1, 2: S0 & ~C2, 3: S0 & ~C3}
            for (a, b) in pair:
                inter = int((R[a] & R[b]).sum())
                union = int((R[a] | R[b]).sum())
                pair[(a, b)][0] += inter
                pair[(a, b)][1] += union
        d['sall'] = None  # 见下：Sall 需逐块累加
        acc[sc] = (d, pair)
        print(f'[indep] {sc} 完成', flush=True)

    # 重新组织（Sall 需累计）
    # 为简洁：第二遍只算 Sall
    for sc in SCENES:
        if sc not in acc:
            continue
        files = sorted(glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')))
        gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
        with rasterio.open(gt_path) as ds:
            gt_full = ds.read(1)
        sall = 0
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
            S0 = s['int'] & (g != 1)
            if not S0.any():
                continue
            post = np.nan_to_num(sar[7], nan=np.nanmedian(sar[7]))
            sall += int((S0 & s['chg'] & (post < POST_THR) & conn_filter(S0, CONN_MIN)).sum())
        acc[sc][0]['sall'] = sall

    L = ['# 验证门约束的假阳集合独立性（实验 A）', '',
         '> 候选池 S0 = S_int ∩ 非洪水（GT≠1），口径同 open 评测。',
         '> 三条约束：C1 = S_chg；C2 = post_VV < −12 dB；C3 = 连通域 ≥500 px（作为独立滤波器作用于 S0）。',
         '> 边际存活率 m_k = |S0∩C_k|/|S0|；独立预测 J_ind = m1·m2·m3。',
         '',
         '| 场景 | S0 假阳 | m1(S_chg) | m2(暗水) | m3(连通域) | **J_ind 预测** | **J_obs 实测** | 实测/预测 |',
         '|:--|--:|--:|--:|--:|--:|--:|--:|']

    tot = dict(s0=0, s1=0, s2=0, s3=0, sall=0)
    for sc in SCENES:
        if sc not in acc:
            continue
        d, _ = acc[sc]
        m1, m2, m3 = d['s1'] / d['s0'], d['s2'] / d['s0'], d['s3'] / d['s0']
        jind = m1 * m2 * m3
        jobs = d['sall'] / d['s0']
        L.append(f'| {sc} | {d["s0"]:,} | {m1:.4f} | {m2:.4f} | {m3:.4f} | '
                 f'{jind:.5f} | **{jobs:.5f}** | {jobs/jind if jind else 0:.2f}× |')
        for k in tot:
            tot[k] += d[k]
    m1, m2, m3 = tot['s1'] / tot['s0'], tot['s2'] / tot['s0'], tot['s3'] / tot['s0']
    jind = m1 * m2 * m3
    jobs = tot['sall'] / tot['s0']
    L.append(f'| **合并** | {tot["s0"]:,} | {m1:.4f} | {m2:.4f} | {m3:.4f} | '
             f'{jind:.5f} | **{jobs:.5f}** | **{jobs/jind:.2f}×** |')
    L.append('')

    L += ['## 约束间假阳剔除集合的两两重叠（Jaccard）', '',
          '| 场景 | J(C1,C2) | J(C1,C3) | J(C2,C3) |', '|:--|--:|--:|--:|']
    for sc in SCENES:
        if sc not in acc:
            continue
        _, pair = acc[sc]
        vals = []
        for k in [(1, 2), (1, 3), (2, 3)]:
            i_, u_ = pair[k]
            vals.append(f'{i_/u_:.3f}' if u_ else '—')
        L.append(f'| {sc} | ' + ' | '.join(vals) + ' |')
    L.append('')

    ratio = jobs / jind if jind else 0
    verdict = ('各约束的假阳集合**近乎独立**，"合取使假阳相乘"成立'
               if 0.8 <= ratio <= 1.25 else
               '实测 > 预测 → 约束**冗余**（剔除的假阳高度重叠），原断言过强'
               if ratio > 1.25 else
               '实测 < 预测 → **次独立**：各约束剔除互不相交的假阳子集，复合优于独立假设')
    L += ['## 判定', '',
          f'- 实测/独立预测 = **{ratio:.2f}×**', f'- {verdict}', '']

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('saved:', OUT, flush=True)
    print(f'J_ind={jind:.5f}  J_obs={jobs:.5f}  ratio={ratio:.2f}', flush=True)


if __name__ == '__main__':
    main()
