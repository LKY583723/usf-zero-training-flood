#!/usr/bin/env python3
"""Phase 2 · 城区分支测试集验证（class-aware GT==2，检测语义）。

目标：把 Phase 1 校准选出的城区分支（U）接到 config F 上形成双类输出 F|U，
在 4 个测试场景上用与表 II 相同的"检测语义"（GT==2 正类、任一洪水预测为正）
评测城区 F1；同时量化 U 分支对 open 口径的影响。

协议：testing_case_256、保留 (g>0)>=100 的 patch、像素聚合、Jubba 混淆矩阵合并。
与 _eval_3event.py / _eval_urban_table.py 完全一致。

产出：docs/03_结果/flood_ge_urban_branch_eval.md
"""
import glob
import os
import sys

_REPRO = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_REPRO)
sys.path.insert(0, _REPRO)

import numpy as np
import rasterio
from flood_strategies import (all_strategies, conn_filter,
                              CO_PRE_VV, CO_CO_VV, CO_PRE_VH, CO_CO_VH)

DATA = os.environ.get('USF_ROOT', _DATA)
PATCHES = os.path.join(DATA, 'testing_case_256')
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
CONN_MIN = 500
POST_THR = -12.0

# Phase 1 校准选出的候选城区分支（P,D,pm），pol 策略：union / vh
URBAN_CANDIDATES = [
    ('U_union_pm3_p6_d3',  'union', 0.6, 0.30, -3.0),
    ('U_vh_pm6_p6_d3',     'vh',    0.6, 0.30, -6.0),
    ('U_vh_pm3_p6_d25',    'vh',    0.6, 0.25, -3.0),
]


def parse_id(fn):
    import re
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def urban_mask(sar, pol, P, D, pm):
    """确定性城区分支：灾前高相干区的大幅去相干 + 灾后 VV 亮回波。"""
    def finite(a):
        return np.nan_to_num(a, nan=np.nanmedian(a))
    if pol == 'union':
        dvv = finite(sar[CO_PRE_VV]) - finite(sar[CO_CO_VV])
        pvv = finite(sar[CO_PRE_VV])
        dvh = finite(sar[CO_PRE_VH]) - finite(sar[CO_CO_VH])
        pvh = finite(sar[CO_PRE_VH])
        m = ((dvv > D) & (pvv > P)) | ((dvh > D) & (pvh > P))
    else:  # vh
        dvh = finite(sar[CO_PRE_VH]) - finite(sar[CO_CO_VH])
        pvh = finite(sar[CO_PRE_VH])
        m = (dvh > D) & (pvh > P)
    post = finite(sar[7])
    return m & (post > pm)


def config_f_mask(sar):
    s = all_strategies(sar)
    f = s['int'] & s['chg']
    f = f & (np.nan_to_num(sar[7]) < POST_THR)
    f = conn_filter(f, CONN_MIN)
    return f


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


def main():
    names = ['F'] + [c[0] for c in URBAN_CANDIDATES]
    # acc[scene][name]['urban'|'open'|'open_neutral'] = [tp, fp, fn]
    acc = {}
    urban_gt_px = {}
    for sc in SCENES:
        patch_dir = os.path.join(PATCHES, sc)
        files = sorted(glob.glob(os.path.join(patch_dir, '*_SAR.tif')))
        gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
        if not (files and os.path.exists(gt_path)):
            continue
        with rasterio.open(gt_path) as ds:
            gt_full = ds.read(1)
        acc[sc] = {m: {'urban': [0, 0, 0], 'open': [0, 0, 0], 'open_neutral': [0, 0, 0]}
                   for m in names}
        urban_gt_px[sc] = 0
        for f in files:
            r, c = parse_id(f)
            y0, x0 = r * 256, c * 256
            if y0 + 256 > gt_full.shape[0] or x0 + 256 > gt_full.shape[1]:
                continue
            g = gt_full[y0:y0 + 256, x0:x0 + 256]
            if (g > 0).sum() < 100:
                continue
            urban_gt_px[sc] += int((g == 2).sum())
            with rasterio.open(f) as ds:
                sar = ds.read().astype(np.float32)
            fm = config_f_mask(sar)
            masks = {'F': fm}
            for name, pol, P, D, pm in URBAN_CANDIDATES:
                masks[name] = fm | urban_mask(sar, pol, P, D, pm)
            # urban 检测语义：正类 GT==2
            u = (g == 2)
            # open：正类 GT==1；open_neutral：排除 g==2 像素（评估"城区GT压低 open 多少"）
            g_n = np.where(g == 2, 0, g)
            for m, pred in masks.items():
                a = acc[sc][m]
                a['urban'][0] += int((pred & u).sum())
                a['urban'][1] += int((pred & ~u).sum())
                a['urban'][2] += int((~pred & u).sum())
                a['open'][0] += int((pred & (g == 1)).sum())
                a['open'][1] += int((pred & (g != 1)).sum())
                a['open'][2] += int((~pred & (g == 1)).sum())
                a['open_neutral'][0] += int((pred & (g_n == 1)).sum())
                a['open_neutral'][1] += int((pred & (g_n == 0)).sum())
                a['open_neutral'][2] += int((~pred & (g_n == 1)).sum())

    # 输出 md
    lines = ['# 城区分支测试集验证（class-aware GT==2 · 检测语义 F|U）', '']
    lines.append('> 协议：testing_case_256，保留 (g>0)>=100，像素聚合，Jubba 混淆矩阵合并。')
    lines.append('> 候选 U：Phase 1 校准（官方 train+valid）选出的 top 设置——阈值不在测试集上选。')
    lines.append('> F = config F（open 主口径不变）；F|U = F 与城区分支的并集掩膜（双类输出）。')
    lines.append('')

    for sc in SCENES:
        lines.append(f'## {sc}')
        lines.append('')
        lines.append(f'城区 GT 像素（含入 patch 内）: {urban_gt_px[sc]}')
        lines.append('')
        lines.append('| 配置 | 城区F1 | P | R | TP | FP | FN | openF1(标准) | openF1(城区中性) |')
        lines.append('|:--|:--:|:--:|:--:|--:|--:|--:|:--:|:--:|')
        for m in names:
            a = acc[sc][m]
            uf = f1(*a['urban']); of = f1(*a['open']); on = f1(*a['open_neutral'])
            pu = a['urban'][0] / (a['urban'][0] + a['urban'][1]) if a['urban'][0] + a['urban'][1] else 0
            ru = a['urban'][0] / (a['urban'][0] + a['urban'][2]) if a['urban'][0] + a['urban'][2] else 0
            lines.append(f'| {m} | {uf:.3f} | {pu:.3f} | {ru:.3f} | {a["urban"][0]} | {a["urban"][1]} | {a["urban"][2]} | {of:.3f} | {on:.3f} |')
        lines.append('')

    # Jubba 合并 + 3 事件均值
    lines.append('## Jubba 合并（Jubba_1 + Jubba_2 混淆矩阵相加）与 3 事件均值')
    lines.append('')
    lines.append('| 配置 | Jubba 城区F1 | 3事件城区F1均值 | 3事件 openF1均值 |')
    lines.append('|:--|:--:|:--:|:--:|')
    for m in names:
        jubba = {'urban': [0, 0, 0], 'open': [0, 0, 0]}
        for k in ('urban', 'open'):
            for i in range(3):
                jubba[k][i] = (acc['20231201_Jubba_1'][m][k][i]
                               + acc['20231201_Jubba_2'][m][k][i])
        jf = f1(*jubba['urban'])
        evs = [acc['20210727_Weihui'][m], acc['20230609_NovaKakhovka'][m], jubba]
        muf = float(np.mean([f1(*e['urban']) for e in evs]))
        mof = float(np.mean([f1(*e['open']) for e in evs]))
        lines.append(f'| {m} | {jf:.3f} | {muf:.3f} | {mof:.3f} |')
    lines.append('')

    # open 口径交互诊断：城区 GT 被 F 预测（= open FP 的来源）
    lines.append('## 诊断：城区 GT 像素对 open F1 的压制（F 掩膜）')
    lines.append('')
    lines.append('open 标准口径把 GT==2 且被预测的像素计为 open 的 FP；"城区中性"把 g==2 排除在混淆外。差值=城区类的压制量。')
    lines.append('')
    lines.append('| 场景 | openF1 标准 | openF1 城区中性 | 差值 |')
    lines.append('|:--|:--:|:--:|:--:|')
    for sc in SCENES:
        a = acc[sc]['F']
        of = f1(*a['open']); on = f1(*a['open_neutral'])
        lines.append(f'| {sc} | {of:.3f} | {on:.3f} | {on - of:+.3f} |')
    lines.append('')

    out = 'D:/DisasterM3/UrbanSARFloods/docs/03_结果/flood_ge_urban_branch_eval.md'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved:', out, flush=True)


if __name__ == '__main__':
    main()
