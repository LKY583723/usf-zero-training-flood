#!/usr/bin/env python3
"""Phase 2b · 城区分支 + 建成区空间先验（测试集）。

对比：
  U_base  = 旧城区分支最优点（无空间约束）P=0.6 D=0.30 pm=-3     ← 基线
  U_abs   = U_base ∩ 建成区掩膜(绝对阈值 tex>3.0dB, gpre>0.4, CC≥500)
            ← Phase 1b 校准集 argmax，**预注册候选**
  U_pct   = U_base ∩ 建成区掩膜(逐场景 p90 分位数阈值, gpre>0.4, CC≥500)
            ← 探索性：诊断显示绝对阈值在 Jubba 覆盖率→0，故试自适应版

口径与 _eval_urban_branch.py 完全一致：testing_case_256、(g>0)>=100 过滤、
像素聚合、Jubba 混淆矩阵合并、城区检测语义（GT==2 为正，任一预测覆盖即为检出）。

注意：城区 FP 包含"正确地检出了开放水体"的像素（与表 III 同口径）。

产出：docs/03_结果/flood_ge_urban_spatial_eval.md
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
from flood_strategies import (CO_PRE_VV, CO_CO_VV, CO_PRE_VH, CO_CO_VH,
                              INT_PRE_VV, conn_filter)
from _calib_urban_spatial import local_std, TEX_WIN

DATA = os.environ.get('USF_ROOT', _DATA)
PATCHES = os.path.join(DATA, 'testing_case_256')
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']

P_FIX, D_FIX, PM_FIX = 0.6, 0.30, -3.0
TEX_ABS, PRE_COH_BU, BU_AMIN = 3.0, 0.4, 500
PCT_ADAPT = 90          # 逐场景纹理分位数

OUT = os.path.join(_REPRO, os.pardir, 'docs', '03_结果',
                   'flood_ge_urban_spatial_eval.md')


def _fin(a):
    m = np.nanmedian(a)
    return np.nan_to_num(a, nan=(m if np.isfinite(m) else 0.0))


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def base_urban(sar):
    pvv, cvv = _fin(sar[CO_PRE_VV]), _fin(sar[CO_CO_VV])
    pvh, cvh = _fin(sar[CO_PRE_VH]), _fin(sar[CO_CO_VH])
    m = (((pvv - cvv) > D_FIX) & (pvv > P_FIX)) | \
        (((pvh - cvh) > D_FIX) & (pvh > P_FIX))
    return m & (_fin(sar[7]) > PM_FIX)


def bu_mask(tex, gpre, thr):
    m = (tex > thr) & (gpre > PRE_COH_BU)
    if not m.any():
        return m
    import cv2
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((5, 5), np.uint8)).astype(bool)
    return conn_filter(m, min_size=BU_AMIN)


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0


def scene_files(sc):
    return sorted(glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')))


def main():
    names = ['F', 'U_base', 'U_abs', 'U_pct']
    acc = {}
    gt_px = {}
    cov = {}

    for sc in SCENES:
        files = scene_files(sc)
        gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
        if not (files and os.path.exists(gt_path)):
            print(f'[skip] {sc}', flush=True)
            continue
        with rasterio.open(gt_path) as ds:
            gt_full = ds.read(1)

        # ---- Pass 1：收集该场景纹理分位数（只读 SAR，不用 GT）----
        pool = []
        for f in files:
            r, c = parse_id(f)
            y0, x0 = r * 256, c * 256
            if y0 + 256 > gt_full.shape[0] or x0 + 256 > gt_full.shape[1]:
                continue
            with rasterio.open(f) as ds:
                sar = ds.read().astype(np.float32)
            pool.append(local_std(_fin(sar[INT_PRE_VV]), TEX_WIN)[::2, ::2].ravel())
        pct_thr = float(np.percentile(np.concatenate(pool), PCT_ADAPT)) if pool else TEX_ABS

        # ---- Pass 2：评测 ----
        acc[sc] = {m: [0, 0, 0] for m in names}
        gt_px[sc] = 0
        cov[sc] = {'abs': [0, 0], 'pct': [0, 0]}
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
            u = (g == 2)
            gt_px[sc] += int(u.sum())

            b = base_urban(sar)
            tex = local_std(_fin(sar[INT_PRE_VV]), TEX_WIN)
            gpre = _fin(sar[CO_PRE_VV])
            ba = bu_mask(tex, gpre, TEX_ABS)
            bp = bu_mask(tex, gpre, pct_thr)
            cov[sc]['abs'][0] += int(ba.sum()); cov[sc]['abs'][1] += ba.size
            cov[sc]['pct'][0] += int(bp.sum()); cov[sc]['pct'][1] += bp.size

            preds = {'U_base': b, 'U_abs': b & ba, 'U_pct': b & bp, 'F': None}
            for m, pred in preds.items():
                if pred is None:
                    continue
                a = acc[sc][m]
                a[0] += int((pred & u).sum())
                a[1] += int((pred & ~u).sum())
                a[2] += int((~pred & u).sum())
        print(f'[eval] {sc} 完成, p{PCT_ADAPT} 阈值={pct_thr:.2f} dB', flush=True)

    # ---- 输出 ----
    L = ['# 城区分支 + 建成区空间先验 · 测试集验证', '',
         '> 协议：testing_case_256、(g>0)>=100、像素聚合、Jubba 混淆矩阵合并。',
         '> 城区检测语义（GT==2 为正，任一预测覆盖即为检出）——FP 含"正确地检出开放水体"的像素，与表 III 同口径。',
         '> **U_abs 是 Phase 1b 校准集 argmax，预先选定，未在测试集上挑选**；U_pct 为探索性自适应版。',
         '']

    for sc in SCENES:
        if sc not in acc:
            continue
        a = acc[sc]
        L += [f'## {sc}', '', f'城区 GT 像素: {gt_px[sc]}', '',
              f'建成区掩膜覆盖率：绝对阈值 {cov[sc]["abs"][0]/max(cov[sc]["abs"][1],1):.4f}'
              f' / 逐场景分位数 {cov[sc]["pct"][0]/max(cov[sc]["pct"][1],1):.4f}', '',
              '| 配置 | 城区F1 | 精度 | 召回 | TP | FP | FN |', '|:--|--:|--:|--:|--:|--:|--:|']
        for m in names:
            if m not in a:
                continue
            tp, fp, fn = a[m]
            L.append(f'| {m} | {f1(tp, fp, fn):.4f} | '
                     f'{tp/(tp+fp) if tp+fp else 0:.4f} | '
                     f'{tp/(tp+fn) if tp+fn else 0:.4f} | {tp} | {fp} | {fn} |')
        L.append('')

    # 3 事件均值（Jubba 合并）
    L += ['## 汇总（3 事件均值）', '', '| 配置 | 城区F1 均值 |', '|:--|--:|']
    for m in names:
        vals = [f1(*acc[sc][m]) for sc in SCENES if sc in acc and m in acc[sc]]
        if vals:
            L.append(f'| {m} | {np.mean(vals):.4f} |')
    L.append('')
    L += ['## 判定', '',
          '- 预注册候选为 **U_abs**；U_pct 仅作机制诊断，不作为论文主结果。', '']

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('saved:', OUT, flush=True)
    for sc in SCENES:
        if sc in acc:
            for m in names:
                if m in acc[sc]:
                    print(f'  {sc:24s} {m:7s} F1={f1(*acc[sc][m]):.4f}', flush=True)


if __name__ == '__main__':
    main()
