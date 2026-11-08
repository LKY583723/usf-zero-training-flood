"""在官方 train+valid 划分上校准 GE-F 的固定阈值（POST_THR, CONN_MIN）。

协议：与测试评测完全一致 —— 256² 象限（512² 训练 chip 切 4 象限），
保留 (g>0)≥100 的象限，像素聚合 open F1（GT==1）。
选择标准：校准集上的均值 open F1（train+valid 合并）。

产出：F1 网格 → 选定 (POST_THR, CONN_MIN) → 供测试事件评测。
"""
import os
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import rasterio
from data_loader import parse_split
from flood_strategies import (strategy_int, strategy_chg, combine_pol,
                              conn_filter, INT_POST_VV)

ROOT = 'D:/DisasterM3/UrbanSARFloods/urban_sar_floods'
OUT = 'D:/DisasterM3/experiment1/results/flood_ge_threshold_calibration.md'

POST_THRESH = [None, -15, -14, -13, -12, -11, -10]
CONN_SIZES = [500, 800, 1200, 1500, 2000]


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


def main():
    samples = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')
    print(f'[calib] 样本总数: {len(samples)}', flush=True)

    # 混淆矩阵累计: key=(thr_idx, conn) -> [tp,fp,fn] for open/all
    acc = {}
    n_quad = 0
    for i, (sar_p, gt_p) in enumerate(samples):
        with rasterio.open(gt_p) as ds:
            g = ds.read(1)
        H, W = g.shape
        # 512 -> 4 个 256 象限
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
                post = np.nan_to_num(sar[INT_POST_VV])
                n_quad += 1
                for ti, thr in enumerate(POST_THRESH):
                    m = base if thr is None else base & (post < thr)
                    for conn in CONN_SIZES:
                        cm = conn_filter(m, conn)
                        key = (ti, conn)
                        a = acc.setdefault(key, {'open': [0, 0, 0], 'all': [0, 0, 0]})
                        a['open'][0] += (cm & (q == 1)).sum()
                        a['open'][1] += (cm & (q != 1)).sum()
                        a['open'][2] += (~cm & (q == 1)).sum()
                        a['all'][0] += (cm & (q > 0)).sum()
                        a['all'][1] += (cm & (q == 0)).sum()
                        a['all'][2] += (~cm & (q > 0)).sum()
        if (i + 1) % 300 == 0:
            print(f'  [calib] {i+1}/{len(samples)} 样本, {n_quad} 洪水象限', flush=True)

    print(f'[calib] 洪水象限总计: {n_quad}', flush=True)

    # 输出网格
    lines = [f'# GE-F 阈值校准（官方 train+valid，{n_quad} 个 256² 洪水象限）', '',
             '> 口径：256² 象限像素聚合 open F1（GT==1），与测试评测一致。', '',
             '## open F1 网格', '',
             '| POST_THR | 500 | 800 | 1200 | 1500 | 2000 |', '|:--|:--:|:--:|:--:|:--:|:--:|']
    rows = []
    for ti, thr in enumerate(POST_THRESH):
        cells = [str(thr) if thr is not None else 'None']
        for conn in CONN_SIZES:
            a = acc[(ti, conn)]
            v = f1(*a['open'])
            cells.append(f'{v:.3f}')
        rows.append((thr, cells))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')

    # 最优组合（open F1）
    best = max(acc.items(), key=lambda kv: f1(*kv[1]['open']))
    (ti_b, conn_b), a_b = best
    thr_b = POST_THRESH[ti_b]
    lines.append(f'## 选定设置（校准集 open F1 最优）')
    lines.append(f'- **POST_THR = {thr_b}, CONN_MIN = {conn_b}**')
    lines.append(f'- 校准集 open F1 = {f1(*a_b["open"]):.3f}, all F1 = {f1(*a_b["all"]):.3f}')
    lines.append('')
    lines.append('## 其他候选')
    ranked = sorted(acc.items(), key=lambda kv: -f1(*kv[1]['open']))[:10]
    for (ti, conn), a in ranked:
        lines.append(f'- thr={POST_THRESH[ti]!s:5s} conn={conn}: open F1={f1(*a["open"]):.3f} all F1={f1(*a["all"]):.3f}')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved:', OUT, flush=True)
    print('best:', thr_b, conn_b, 'open F1=', round(f1(*a_b['open']), 3), flush=True)


if __name__ == '__main__':
    main()
