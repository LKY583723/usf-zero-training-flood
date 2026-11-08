#!/usr/bin/env python3
"""诊断 · 建成区先验的绝对阈值能否迁移到测试场景。

问题：Phase 1b 的建成区掩膜用**绝对**纹理阈值（tex > T_dB）。若测试场景的
纹理统计与校准图块系统性偏移，该阈值无法迁移——这正是旧城区分支失败的同
一类原因（固定阈值 vs 场景分布漂移）。

本脚本**只读 SAR 波段**（灾前 VV 强度 band5、灾前 VV 相干 band1），
**不使用任何 GT**，因此不消耗测试口径纪律。

输出：docs/03_结果/flood_ge_urban_spatial_transfer.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rasterio
import cv2
from data_loader import parse_split
from flood_strategies import CO_PRE_VV, INT_PRE_VV, conn_filter
from _calib_urban_spatial import local_std, build_bu_mask, TEX_WIN

_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
# 两个 root：数据集（Train/Valid_dataset.txt + 01_NF/02_FO/03_FU）在 urban_sar_floods/，
# 测试 patch 在仓库根目录的 testing_case_256/。
_DATA_ROOT = os.path.abspath(os.path.join(_REPRO_DIR, os.pardir, 'urban_sar_floods'))
_TEST_ROOT = os.path.abspath(os.path.join(_REPRO_DIR, os.pardir))
ROOT = os.environ.get('USF_ROOT', _DATA_ROOT)
PATCHES = os.path.join(os.environ.get('USF_TEST_ROOT', _TEST_ROOT), 'testing_case_256')
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
OUT = os.path.join(_REPRO_DIR, os.pardir, 'docs', '03_结果',
                   'flood_ge_urban_spatial_transfer.md')

TEX_DBS = [2.0, 3.0, 4.0]
BU = (3.0, 0.4, 500, 0)      # Phase 1b 校准最优点
N_CAL_CHIPS = 800
PCTS = [10, 50, 70, 84, 90, 95, 99]


def _fin(a):
    m = np.nanmedian(a)
    if not np.isfinite(m):
        m = 0.0
    return np.nan_to_num(a, nan=m)


def stats_from_chips(paths):
    """返回 (纹理分位数, 各 T_dB 超阈比例, 建成区覆盖率, 有效像素数)。"""
    tex_pool = []
    exceed = {t: 0 for t in TEX_DBS}
    cover = 0
    n_px = 0
    for p in paths:
        with rasterio.open(p) as ds:
            sar = ds.read().astype(np.float32)
        if sar.shape[1] < 64 or sar.shape[2] < 64:
            continue
        gpre = _fin(sar[CO_PRE_VV])
        tex = local_std(_fin(sar[INT_PRE_VV]), TEX_WIN)
        valid = np.isfinite(tex)
        n_px += int(valid.sum())
        # 分位数用子采样以控内存
        sub = tex[::4, ::4]
        tex_pool.append(sub)
        for t in TEX_DBS:
            exceed[t] += int((tex > t).sum())
        bu = build_bu_mask(tex, gpre, *BU)
        if bu is not None:
            cover += int(bu.sum())
    pool = np.concatenate([a.ravel() for a in tex_pool])
    return (np.percentile(pool, PCTS),
            {t: exceed[t] / max(n_px, 1) for t in TEX_DBS},
            cover / max(n_px, 1), n_px)


def main():
    cal = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')
    rng = np.random.default_rng(0)
    idx = rng.choice(len(cal), size=min(N_CAL_CHIPS, len(cal)), replace=False)
    cal_paths = [cal[i][0] for i in idx]
    print(f'[diag] 校准图块采样 {len(cal_paths)}', flush=True)
    cal_q, cal_ex, cal_cov, cal_n = stats_from_chips(cal_paths)
    print('[diag] 校准集完成', flush=True)

    res = {}
    for sc in SCENES:
        d = os.path.join(PATCHES, sc)
        paths = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.lower().endswith(('.tif', '.tiff', '.png'))]
        q, ex, cov, n = stats_from_chips(paths)
        res[sc] = (q, ex, cov, n)
        print(f'[diag] {sc} 完成 ({len(paths)} patch)', flush=True)

    L = [
        '# 建成区先验 · 绝对阈值可迁移性诊断（仅用 SAR，不用 GT）',
        '',
        f'> 纹理 = 灾前 VV 强度(dB) 的 {TEX_WIN}×{TEX_WIN} 局部标准差；窗口与 Phase 1b 一致。',
        f'> 校准集采样 {len(cal_paths)} 个官方 train/valid 图块；测试场景各 {len(os.listdir(os.path.join(PATCHES, SCENES[0])))} patch。',
        '> **本诊断不读任何 GT**，因此不构成对测试集的调参。',
        '',
        '## 纹理分位数（dB）',
        '',
        '| 来源 | ' + ' | '.join(f'p{p}' for p in PCTS) + ' |',
        '|:--|' + '--:|' * len(PCTS),
        '| **校准集** | ' + ' | '.join(f'{v:.2f}' for v in cal_q) + ' |',
    ]
    for sc in SCENES:
        L.append(f'| {sc} | ' + ' | '.join(f'{v:.2f}' for v in res[sc][0]) + ' |')
    L.append('')

    L += ['## 超阈像素比例（建成区掩膜的原料）', '',
          '| 来源 | tex>2.0dB | tex>3.0dB | tex>4.0dB | 建成区掩膜覆盖率 |',
          '|:--|--:|--:|--:|--:|',
          f'| **校准集** | ' + ' | '.join(f'{cal_ex[t]:.3f}' for t in TEX_DBS) +
          f' | {cal_cov:.3f} |']
    for sc in SCENES:
        _, ex, cov, _ = res[sc]
        L.append(f'| {sc} | ' + ' | '.join(f'{ex[t]:.3f}' for t in TEX_DBS) +
                 f' | {cov:.3f} |')
    L.append('')

    # 判定
    cal_p90 = cal_q[PCTS.index(90)]
    L += ['## 判定', '']
    worst = 0.0
    for sc in SCENES:
        q = res[sc][0]
        d = q[PCTS.index(90)] - cal_p90
        worst = max(worst, abs(d))
        L.append(f'- {sc}: p90 纹理 = {q[PCTS.index(90)]:.2f} dB，'
                 f'相对校准集 {cal_p90:.2f} dB 偏移 {d:+.2f} dB')
    L.append('')
    L.append(f'- 最大 p90 偏移: **{worst:.2f} dB**')
    L.append('- 判据：偏移 < 1 dB → 绝对阈值可迁移，值得跑完整测试集；')
    L.append('  偏移 ≥ 1 dB → 绝对阈值不可靠，应改用逐场景分位数阈值。')
    L.append('')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('saved:', OUT, flush=True)
    for sc in SCENES:
        q = res[sc][0]
        print(f'{sc}: p90={q[PCTS.index(90)]:.2f} cov={res[sc][2]:.3f}', flush=True)


if __name__ == '__main__':
    main()
