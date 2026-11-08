"""Reviewer response M1: is the gate, or the threshold granularity, what matters?

Adds a split-based (block-wise Otsu) baseline to the same 3-event protocol used for
configuration F:

  A     per-scene Otsu dark water, no gate                (published: 0.218 mean)
  S     block-wise Otsu dark water, no gate               (split-based operational style)
  S+F   block-wise Otsu + the two fixed constants         (same gate as F)
  F     per-scene Otsu intersection + gate                (published: 0.654 mean)

Everything else — patch filter (>=100 flood pixels), pixel aggregation, Jubba merged by
summing confusion matrices — is identical to _eval_3event.py.
"""
import glob
import os
import re
import sys

_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.dirname(_REPRO_DIR)
sys.path.insert(0, _REPRO_DIR)
os.environ['USF_NORM'] = 'minmax'

import numpy as np
import rasterio
from flood_strategies import (INT_POST_VV, INT_POST_VH, _otsu_binary,
                              all_strategies, conn_filter)

DATA = os.environ.get('USF_ROOT', _DATA_ROOT)
PATCHES = os.path.join(DATA, 'testing_case_256')
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
CONN_MIN, POST_THR = 500, -12.0
BLOCKS = (16, 32, 64, 128)


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def block_otsu(channel, block):
    """Split-based thresholding: independent Otsu inside every block x block tile."""
    out = np.zeros(channel.shape, dtype=bool)
    h, w = channel.shape
    for y in range(0, h, block):
        for x in range(0, w, block):
            out[y:y + block, x:x + block] = _otsu_binary(channel[y:y + block, x:x + block], flip=True)
    return out


def dark_blocks(sar, block):
    """Block-wise dark-water mask, both polarizations unioned (as configuration F does)."""
    return block_otsu(sar[INT_POST_VV], block) | block_otsu(sar[INT_POST_VH], block)


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0


NAMES = ['A', 'F'] + [f'S{b}' for b in BLOCKS] + [f'S{b}+F' for b in BLOCKS]
acc = {}
for sc in SCENES:
    files = sorted(glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')))
    gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
    if not (files and os.path.exists(gt_path)):
        continue
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)
    acc[sc] = {m: [0, 0, 0] for m in NAMES}          # tp, fp, fn on the open class
    for path in files:
        r, c = parse_id(path)
        y0, x0 = r * 256, c * 256
        if y0 + 256 > gt_full.shape[0] or x0 + 256 > gt_full.shape[1]:
            continue
        g = gt_full[y0:y0 + 256, x0:x0 + 256]
        if (g > 0).sum() < 100:
            continue
        with rasterio.open(path) as ds:
            sar = ds.read().astype(np.float32)
        s = all_strategies(sar)
        preds = {'A': s['int']}
        gate = conn_filter((np.nan_to_num(sar[7]) < POST_THR), CONN_MIN)
        preds['F'] = conn_filter(s['int'] & s['chg'] & (np.nan_to_num(sar[7]) < POST_THR), CONN_MIN)
        for b in BLOCKS:
            sb = dark_blocks(sar, b)
            preds[f'S{b}'] = sb
            preds[f'S{b}+F'] = conn_filter(sb & (np.nan_to_num(sar[7]) < POST_THR), CONN_MIN)
        for m, pred in preds.items():
            acc[sc][m][0] += int((pred & (g == 1)).sum())
            acc[sc][m][1] += int((pred & (g != 1)).sum())
            acc[sc][m][2] += int((~pred & (g == 1)).sum())

jubba = {m: [sum(acc[s][m][i] for s in ('20231201_Jubba_1', '20231201_Jubba_2')) for i in range(3)]
         for m in NAMES}

print('=' * 74)
print('M1 · threshold granularity vs the gate (3-event protocol, open F1)')
print('=' * 74)
print(f'  {"config":10s} {"Weihui":>8s} {"Nova":>8s} {"Jubba":>8s} {"mean":>8s}')
for m in NAMES:
    evs = [acc['20210727_Weihui'][m], acc['20230609_NovaKakhovka'][m], jubba[m]]
    vals = [f1(*e) for e in evs]
    print(f'  {m:10s} {vals[0]:8.3f} {vals[1]:8.3f} {vals[2]:8.3f} {np.mean(vals):8.3f}')
