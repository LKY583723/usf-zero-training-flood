#!/usr/bin/env python3
"""验证 2 · 城区分支逐事件表现的可视化抽检。

取一个高分事件（20180508_Somalia, F1 0.619）与一个低分事件
（20160419_Houston, F1 0.041），各选城区 GT 最多的象限，
并排显示：灾后 VV 强度 / GT(城区) / 分支预测。

目的：确认高 F1 不是标注或解析假象——分支应当确实勾勒出城区洪水范围；
低分事件的失效模式应当肉眼可见（假阳打在农田/植被上，或完全够不着）。

产出：docs/03_结果/fig_urban_event_check.png
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import parse_split
from flood_strategies import CO_PRE_VV, CO_CO_VV, CO_PRE_VH, CO_CO_VH

_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.abspath(os.path.join(_REPRO_DIR, os.pardir, 'urban_sar_floods'))
ROOT = os.environ.get('USF_ROOT', _DATA_ROOT)
OUT = os.path.join(_REPRO_DIR, os.pardir, 'docs', '03_结果',
                   'fig_urban_event_check.png')

P_FIX, D_FIX, PM_FIX = 0.6, 0.30, -3.0
TARGETS = ['20180508_Somalia', '20160419_Houston']


def event_of(path):
    m = re.search(r'(\d{8}_[A-Za-z0-9_]+?)_ID_', os.path.basename(path))
    return m.group(1) if m else 'UNKNOWN'


def fin(a):
    m = np.nanmedian(a)
    return np.nan_to_num(a, nan=(m if np.isfinite(m) else 0.0))


def best_quadrant(samples, ev):
    """返回该事件中城区 GT 像素最多的象限及其数据。"""
    best = None
    for sar_p, gt_p in samples:
        if event_of(sar_p) != ev:
            continue
        with rasterio.open(gt_p) as ds:
            g_full = ds.read(1)
        H, W = g_full.shape
        for y0 in range(0, H - 255, 256):
            for x0 in range(0, W - 255, 256):
                q = g_full[y0:y0 + 256, x0:x0 + 256]
                if (q > 0).sum() < 100:
                    continue
                n_urb = int((q == 2).sum())
                if best is None or n_urb > best[0]:
                    with rasterio.open(sar_p) as ds:
                        sar = ds.read(window=((y0, y0 + 256), (x0, x0 + 256))).astype(np.float32)
                    best = (n_urb, q.copy(), sar, os.path.basename(sar_p), y0, x0)
    return best


def main():
    samples = parse_split(ROOT, 'train') + parse_split(ROOT, 'valid')
    fig, axes = plt.subplots(len(TARGETS), 3, figsize=(11, 7.6))
    for r, ev in enumerate(TARGETS):
        b = best_quadrant(samples, ev)
        if b is None:
            print(f'[warn] {ev} 无样本', flush=True)
            continue
        n_urb, g, sar, fn, y0, x0 = b
        urban = (g == 2)
        pvv, cvv = fin(sar[CO_PRE_VV]), fin(sar[CO_CO_VV])
        pvh, cvh = fin(sar[CO_PRE_VH]), fin(sar[CO_CO_VH])
        pred = ((((pvv - cvv) > D_FIX) & (pvv > P_FIX)) |
                (((pvh - cvh) > D_FIX) & (pvh > P_FIX))) & (fin(sar[7]) > PM_FIX)
        tp = int((pred & urban).sum()); fp = int((pred & ~urban).sum())
        fn_ = int((~pred & urban).sum())
        post = fin(sar[7])
        lo, hi = np.percentile(post, [2, 98])

        axes[r, 0].imshow(post, cmap='gray', vmin=lo, vmax=hi)
        axes[r, 0].set_title(f'{ev}\npost-VV intensity  ({fn} @ {y0},{x0})', fontsize=9)
        axes[r, 1].imshow(post, cmap='gray', vmin=lo, vmax=hi)
        axes[r, 1].imshow(np.ma.masked_where(~urban, urban), cmap='autumn', alpha=0.6)
        axes[r, 1].set_title(f'GT urban (red), {n_urb} px', fontsize=9)
        axes[r, 2].imshow(post, cmap='gray', vmin=lo, vmax=hi)
        axes[r, 2].imshow(np.ma.masked_where(~pred, pred), cmap='winter', alpha=0.6)
        axes[r, 2].set_title(f'branch pred (cyan)\nTP {tp} / FP {fp} / FN {fn_}  '
                             f'P {tp/(tp+fp) if tp+fp else 0:.2f} R {tp/(tp+fn_) if tp+fn_ else 0:.2f}',
                             fontsize=9)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        print(f'{ev}: TP {tp} FP {fp} FN {fn_}', flush=True)

    plt.tight_layout()
    plt.savefig(OUT, dpi=130, bbox_inches='tight')
    print('saved:', OUT, flush=True)


if __name__ == '__main__':
    main()
