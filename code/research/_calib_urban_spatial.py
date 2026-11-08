#!/usr/bin/env python3
"""Phase 1b · 城区分支 + 逐场景建成区空间先验（校准集）。

动机（见 docs/03_结果/flood_ge_urban_branch_calib.md 的失败诊断）：
  旧城区分支 10 个家族**全是逐像素**判据，无任何空间/结构约束；而开放水的
  config F 决定性一跳恰恰来自连通域滤波。诊断结论是"缺建成区空间先验"，
  但该先验从未被真正试过。

本脚本只加**一个**变量：把检测器限制在"灾前建成区代理图"之内，
检测器本身固定在已知最优 (coh_union, P=0.6, D=0.30, post_min=-3)，
以便干净归因。

建成区代理（全部来自同一对 SLC 的灾前波段，零训练、无外部数据）：
  tex   = 灾前 VV 强度（dB）的局部标准差   —— 建筑区亮墙/暗巷/阴影交替，纹理方差高
  gpre  = 灾前 VV 相干度                    —— 建筑为持久散射体，灾前高相干
  建成区掩膜 = close( tex > T_dB  &  gpre > P_bu ) → 连通域 ≥ A_min → 可选膨胀 d

输出：docs/03_结果/flood_ge_urban_spatial_calib.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rasterio
import cv2
from data_loader import parse_split
from flood_strategies import (CO_PRE_VV, CO_CO_VV, CO_PRE_VH, CO_CO_VH,
                              INT_PRE_VV, INT_POST_VV, conn_filter)

_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.abspath(os.path.join(_REPRO_DIR, os.pardir, 'urban_sar_floods'))
ROOT = os.environ.get('USF_ROOT', _DATA_ROOT)
OUT = os.path.join(_REPRO_DIR, os.pardir, 'docs', '03_结果',
                   'flood_ge_urban_spatial_calib.md')

# 检测器固定在旧校准的最优点（coh_union_pm3）
P_FIX, D_FIX, PM_FIX = 0.6, 0.30, -3.0

# 建成区先验扫描轴
TEX_DBS = [0.0, 2.0, 3.0, 4.0]      # 0 = 关闭纹理约束
BU_COHS = [0.0, 0.4]                 # 0 = 关闭灾前相干约束
BU_AMINS = [0, 500, 5000]            # 0 = 不做连通域过滤
BU_DILS = [0, 8]                     # 掩膜膨胀半径（px）

TEX_WIN = 7                          # 纹理窗口（灾前强度 dB 的局部 std）


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0


def local_std(img, win):
    """局部标准差（float32, 忽略 NaN 后置中位数）。"""
    a = np.nan_to_num(img, nan=np.nanmedian(img)).astype(np.float32)
    k = np.ones((win, win), np.uint8)
    m1 = cv2.boxFilter(a, -1, (win, win), normalize=True)
    m2 = cv2.boxFilter(a * a, -1, (win, win), normalize=True)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def build_bu_mask(tex, gpre, t_db, p_bu, a_min, dil):
    """建成区代理掩膜。t_db=0 且 p_bu=0 且 a_min=0 且 dil=0 → 全 True（=无空间约束）。"""
    if t_db <= 0 and p_bu <= 0 and a_min == 0 and dil == 0:
        return None  # 哨兵：无约束
    m = np.ones(tex.shape, dtype=bool)
    if t_db > 0:
        m &= (tex > t_db)
    if p_bu > 0:
        m &= (gpre > p_bu)
    if not m.any():
        return m
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((5, 5), np.uint8)).astype(bool)
    if a_min > 0:
        m = conn_filter(m, min_size=a_min)
    if dil > 0 and m.any():
        m = cv2.dilate(m.astype(np.uint8),
                       np.ones((2 * dil + 1, 2 * dil + 1), np.uint8)).astype(bool)
    return m


def main():
    samples = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')
    print(f'[calib-urbsp] 样本总数: {len(samples)}', flush=True)

    combos = [(t, p, a, d) for t in TEX_DBS for p in BU_COHS
              for a in BU_AMINS for d in BU_DILS]
    acc = {c: [0, 0, 0] for c in combos}   # [tp, fp, fn] over urban class
    # 不限建成区时的全场景混淆（用于量化"建成区掩膜覆盖率"）
    bu_cover = {c: 0 for c in combos}

    n_quad = 0
    n_urban_px = 0
    n_all_px = 0

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
                n_all_px += int(urban.size)

                fin = lambda a: np.nan_to_num(a, nan=np.nanmedian(a))
                pre_vv, co_vv = fin(sar[CO_PRE_VV]), fin(sar[CO_CO_VV])
                pre_vh, co_vh = fin(sar[CO_PRE_VH]), fin(sar[CO_CO_VH])
                gpre = pre_vv                      # 灾前 VV 相干度
                post_vv = fin(sar[INT_POST_VV])
                pre_i_vv = fin(sar[INT_PRE_VV])
                tex = local_std(pre_i_vv, TEX_WIN)

                # 检测器核心（固定参数）
                det = (((pre_vv - co_vv) > D_FIX) & (pre_vv > P_FIX)) | \
                      (((pre_vh - co_vh) > D_FIX) & (pre_vh > P_FIX))
                det &= (post_vv > PM_FIX)

                n_quad += 1
                for c in combos:
                    bu = build_bu_mask(tex, gpre, *c)
                    m = det if bu is None else (det & bu)
                    cm = acc[c]
                    cm[0] += int((m & urban).sum())
                    cm[1] += int((m & ~urban).sum())
                    cm[2] += int((~m & urban).sum())
                    if bu is not None:
                        bu_cover[c] += int(bu.sum())

        if (i + 1) % 300 == 0:
            print(f'  [calib-urbsp] {i+1}/{len(samples)} 样本, {n_quad} 象限', flush=True)

    print(f'[calib-urbsp] 象限: {n_quad}, 城区GT像素: {n_urban_px}', flush=True)

    rows = []
    for c, cm in acc.items():
        tp, fp, fn = cm
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        rows.append((f1(tp, fp, fn), p_, r_, c, cm))
    rows.sort(reverse=True, key=lambda x: x[0])

    base = acc[(0.0, 0.0, 0, 0)]
    base_f1 = f1(*base)
    base_p = base[0] / (base[0] + base[1]) if base[0] + base[1] else 0.0
    base_r = base[0] / (base[0] + base[2]) if base[0] + base[2] else 0.0

    L = [
        '# 城区分支 + 逐场景建成区空间先验（校准集）',
        '',
        '> 协议：官方 train+valid，512 chip → 4×256 象限，保留 (g>0)>=100，像素聚合，目标 g==2。',
        f'> 象限数: {n_quad} | 城区 GT 像素: {n_urban_px}',
        f'> 检测器**固定**在旧校准最优点：P={P_FIX}, D={D_FIX}, post_vv>{PM_FIX} dB（coh_union 家族）。',
        '',
        '## 建成区代理',
        '',
        f'- `tex` = 灾前 VV 强度(dB) 的 {TEX_WIN}×{TEX_WIN} 局部标准差 —— 建筑区纹理方差高',
        '- `gpre` = 灾前 VV 相干度 —— 建筑为持久散射体',
        '- 掩膜 = close(tex > T_dB  &  gpre > P_bu) → 连通域 ≥ A_min → 膨胀 d px',
        '- **全部来自同一对 SLC 的灾前波段**；无学习参数、无外部数据',
        '',
        '## 无空间约束的基线（= 旧分支最优点，用于对照）',
        '',
        f'- 城区 F1 **{base_f1:.3f}**（精度 {base_p:.3f} / 召回 {base_r:.3f}）',
        f'- TP {base[0]} / FP {base[1]} / FN {base[2]}',
        '',
        '## 扫描结果（降序 Top 20）',
        '',
        '| F1 | 精度 | 召回 | T_dB | P_bu | A_min | dilat | TP | FP | FN | 建成区覆盖率 |',
        '|:--:|:--:|:--:|:--:|:--:|--:|:--:|--:|--:|--:|:--:|',
    ]
    for f1v, p_, r_, c, cm in rows[:20]:
        t, pb, a, d = c
        cov = bu_cover[c] / n_all_px if a or t or pb or d else 1.0
        L.append(f'| {f1v:.3f} | {p_:.3f} | {r_:.3f} | {t} | {pb} | {a} | {d} | '
                 f'{cm[0]} | {cm[1]} | {cm[2]} | {cov:.3f} |')
    L.append('')

    # 逐轴边际效应：固定其他轴取基线值，看单轴变化
    L += ['## 单轴边际效应（其余轴 = 无约束/基线）', '']
    L += ['| 轴 | 取值 | 城区 F1 | 精度 | 召回 |', '|:--|:--|:--:|:--:|:--:|']
    for t in TEX_DBS:
        c = (t, 0.0, 0, 0)
        r_ = next(x for x in rows if x[3] == c)
        L.append(f'| T_dB | {t} | {r_[0]:.3f} | {r_[1]:.3f} | {r_[2]:.3f} |')
    for pb in BU_COHS:
        c = (0.0, pb, 0, 0)
        r_ = next(x for x in rows if x[3] == c)
        L.append(f'| P_bu | {pb} | {r_[0]:.3f} | {r_[1]:.3f} | {r_[2]:.3f} |')
    for a in BU_AMINS:
        c = (3.0, 0.4, a, 0)
        r_ = next(x for x in rows if x[3] == c)
        L.append(f'| A_min (T=3,P_bu=0.4) | {a} | {r_[0]:.3f} | {r_[1]:.3f} | {r_[2]:.3f} |')
    for d in BU_DILS:
        c = (3.0, 0.4, 500, d)
        r_ = next(x for x in rows if x[3] == c)
        L.append(f'| dilate (T=3,P_bu=0.4,A=500) | {d} | {r_[0]:.3f} | {r_[1]:.3f} | {r_[2]:.3f} |')
    L.append('')

    best = rows[0]
    L += [
        '## 判定',
        '',
        f'- 基线（无空间约束）: F1 {base_f1:.3f}',
        f'- 最优（含空间先验）: F1 {best[0]:.3f}  @ T_dB={best[3][0]}, P_bu={best[3][1]}, '
        f'A_min={best[3][2]}, dil={best[3][3]}',
        f'- 增益: {best[0] - base_f1:+.3f}',
        '',
        '- 判据：若增益显著（如 +0.05 以上）且精度未崩 → 值得进入测试集验证；',
        '  若无增益 → 建成区先验对校准集无帮助，**不进入测试集**。',
        '',
    ]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('saved:', OUT, flush=True)
    print(f'baseline F1={base_f1:.3f}  best F1={best[0]:.3f}  gain={best[0]-base_f1:+.3f}',
          flush=True)
    print('best combo:', best[3], flush=True)


if __name__ == '__main__':
    main()
