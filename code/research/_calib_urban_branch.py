#!/usr/bin/env python3
"""Phase 1 · 城区分支校准（class-aware，目标 GT==2）。

目的：判断"零训练相干下降检测器"能否在官方 train+valid 上、以可辩护的
固定阈值检测城区洪水（GT==2）。若校准集城区 F1 达到有意义的量级（>=~0.2-0.3
且 precision 不崩塌）→ 方向 A（双类 config）可行，进入 Phase 2 测试集验证。

协议（对齐测试评测，参考 _calib_threshold.py / _eval_urban_table.py）：
  官方 train+valid 512 chip → 4 个 256 象限；保留 (g>0)>=100 的象限；
  真值正类 = g==2；像素聚合 precision/recall/F1；不做 <100 城区过滤（城区正类本身稀疏）。

检测器族（全部确定性、零训练）：
  coh_vv  : (pre_vv − co_vv) > D  &  pre_vv > P
  coh_vh  : (pre_vh − co_vh) > D  &  pre_vh > P
  coh_union               = coh_vv | coh_vh
  coh_union_and_chg       = coh_union & (S_chg_vv | S_chg_vh)   # 要求跨期强度变化
对每个象限只算一次 dvv/dvh/pvv/pvh/chg 掩膜，网格 (P,D) 用比较累积混淆矩阵。

产出：docs/03_结果/flood_ge_urban_branch_calib.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rasterio
from data_loader import parse_split
from flood_strategies import (CO_PRE_VV, CO_CO_VV, CO_PRE_VH, CO_CO_VH,
                              strategy_chg)

ROOT = 'D:/DisasterM3/UrbanSARFloods/urban_sar_floods'
OUT = 'D:/DisasterM3/UrbanSARFloods/docs/03_结果/flood_ge_urban_branch_calib.md'

PRE_PS = [0.4, 0.5, 0.6]
DELTA_DS = [0.15, 0.20, 0.25, 0.30]
# coh_*_pmN = 另加 "灾后 VV 强度 > N dB" 亮度约束（S_urban 代码默认 post_min=-6）
POST_MIN_MAP = {'coh_vv_pm6': -6.0, 'coh_vv_pm3': -3.0,
                'coh_vh_pm6': -6.0, 'coh_vh_pm3': -3.0,
                'coh_union_pm6': -6.0, 'coh_union_pm3': -3.0}
FAMILIES = ['coh_vv', 'coh_vh', 'coh_union', 'coh_union_and_chg'] + list(POST_MIN_MAP)


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


def main():
    samples = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')
    print(f'[calib-urban] 样本总数: {len(samples)}', flush=True)

    # cm[(fam, P, D)] -> [tp, fp, fn]  (urban)
    acc = {}
    n_quad = 0
    n_urban_px = 0
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
                urban = (q == 2)
                n_urban_px += int(urban.sum())
                # 一次性计算各极化 Δγ 与前相干
                def finite(a):
                    return np.nan_to_num(a, nan=np.nanmedian(a))
                pre_vv = finite(sar[CO_PRE_VV]); co_vv = finite(sar[CO_CO_VV])
                pre_vh = finite(sar[CO_PRE_VH]); co_vh = finite(sar[CO_CO_VH])
                dvv = pre_vv - co_vv
                dvh = pre_vh - co_vh
                post_vv = finite(sar[7])
                chg_vv = strategy_chg(sar, 'vv')
                chg_vh = strategy_chg(sar, 'vh')
                chg_union = chg_vv | chg_vh
                n_quad += 1
                for fam in FAMILIES:
                    pm = POST_MIN_MAP.get(fam)
                    for P in PRE_PS:
                        for D in DELTA_DS:
                            if fam == 'coh_vv':
                                m = (dvv > D) & (pre_vv > P)
                            elif fam == 'coh_vh':
                                m = (dvh > D) & (pre_vh > P)
                            elif fam == 'coh_union':
                                m = ((dvv > D) & (pre_vv > P)) | ((dvh > D) & (pre_vh > P))
                            elif fam == 'coh_union_and_chg':
                                m = (((dvv > D) & (pre_vv > P)) | ((dvh > D) & (pre_vh > P))) & chg_union
                            else:
                                # coh_*_pmN：亮度约束 post_vv > pm
                                base = ((dvv > D) & (pre_vv > P)) if fam.startswith('coh_vv_') else \
                                       ((dvh > D) & (pre_vh > P)) if fam.startswith('coh_vh_') else \
                                       (((dvv > D) & (pre_vv > P)) | ((dvh > D) & (pre_vh > P)))
                                m = base & (post_vv > pm)
                            key = (fam, P, D)
                            cm = acc.setdefault(key, [0, 0, 0])
                            cm[0] += int((m & urban).sum())
                            cm[1] += int((m & ~urban).sum())
                            cm[2] += int((~m & urban).sum())
        if (i + 1) % 300 == 0:
            print(f'  [calib-urban] {i+1}/{len(samples)} 样本, {n_quad} 洪水象限', flush=True)

    print(f'[calib-urban] 洪水象限总计: {n_quad}, 城区GT像素: {n_urban_px}', flush=True)

    # 输出 md
    lines = [
        '# 城区分支校准（官方 train+valid · class-aware GT==2）',
        '',
        f'> 协议：512 chip → 4×256 象限，保留 (g>0)>=100，像素聚合，目标 g==2。',
        f'> 象限数: {n_quad} | 城区 GT 像素总数: {n_urban_px}（小样本预警：像素少时 F1 噪声大）',
        '',
        '## 网格说明',
        '',
        '- **P** = 灾前相干度门槛（pre_coh > P）；**D** = 相干下降门槛（Δγ = pre_coh − co_coh > D）',
        '- coh_vv / coh_vh = 单极化；coh_union = 双极化并集；coh_union_and_chg = 并集 ∩ 跨期强度变化',
        '- 城区洪水先验：建筑区灾前高相干、被淹后去相干；强度为亮双反弹（与开放暗水互斥）',
        '',
    ]
    # 汇总排序表
    rows = []
    for key, cm in acc.items():
        fam, P, D = key
        rows.append((f1(*cm), fam, P, D, cm))
    rows.sort(reverse=True, key=lambda r: r[0])

    lines.append('## 各设置城区 F1（降序 Top 25）')
    lines.append('')
    lines.append('| F1 | P(精度) | R(召回) | 家族 | P阈值 | D阈值 | TP | FP | FN |')
    lines.append('|:--:|:--:|:--:|:--|:--:|:--:|--:|--:|--:|')
    for f1v, fam, P, D, cm in rows[:25]:
        p_ = cm[0] / (cm[0] + cm[1]) if cm[0] + cm[1] else 0
        r_ = cm[0] / (cm[0] + cm[2]) if cm[0] + cm[2] else 0
        lines.append(f'| {f1v:.3f} | {p_:.3f} | {r_:.3f} | {fam} | {P} | {D} | {cm[0]} | {cm[1]} | {cm[2]} |')
    lines.append('')

    lines.append('## 按家族：最佳设置')
    lines.append('')
    lines.append('| 家族 | 最佳(P,D) | F1 | P | R |')
    lines.append('|:--|:--|:--:|:--:|:--:|')
    for fam in FAMILIES:
        fam_rows = [r for r in rows if r[1] == fam]
        if fam_rows:
            best = fam_rows[0]
            p_ = best[4][0] / (best[4][0] + best[4][1]) if best[4][0] + best[4][1] else 0
            r_ = best[4][0] / (best[4][0] + best[4][2]) if best[4][0] + best[4][2] else 0
            lines.append(f'| {fam} | P={best[2]} D={best[3]} | {best[0]:.3f} | {p_:.3f} | {r_:.3f} |')
    lines.append('')

    # 基线参考：S_urban 代码默认(0.6/0.25)与论文(0.5/0.2)在 coh_union 上的位置
    lines.append('## 论文/代码常数的定位（coh_union 家族内）')
    lines.append('')
    lines.append('| (P,D) | F1 | P | R |')
    lines.append('|:--|:--:|:--:|:--:|')
    for P, D in [(0.5, 0.2), (0.6, 0.25)]:
        cm = acc[('coh_union', P, D)]
        p_ = cm[0] / (cm[0] + cm[1]) if cm[0] + cm[1] else 0
        r_ = cm[0] / (cm[0] + cm[2]) if cm[0] + cm[2] else 0
        lines.append(f'| ({P}, {D}) | {f1(*cm):.3f} | {p_:.3f} | {r_:.3f} |')
    lines.append('')

    # 决策提示
    best = rows[0]
    lines.append('## 决策提示')
    lines.append('')
    lines.append(f'- 校准集最优: {best[1]} P={best[2]} D={best[3]} → 城区 F1 {best[0]:.3f}')
    lines.append('- 经验判据（方向 A 可行性）：F1 >= ~0.2–0.3 且 precision 不崩塌（>= ~0.1）→ 进入 Phase 2 测试集验证；否则走分析深度（B）。')
    lines.append('')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved:', OUT, flush=True)
    print('best:', best[1], 'P=', best[2], 'D=', best[3], 'urbanF1=', round(best[0], 3), flush=True)


if __name__ == '__main__':
    main()
