"""VLM 语义 Verifier：Qwen 判断每个 tile 的 flood_visible/mask_quality，
验证"语义门"（False→清空）能否提升 F1。"""
import glob
import os
import re
import sys
import tempfile

import numpy as np
import rasterio
from flood_ge import process_tile, load_prompt, _save_png, _mask_overlay, LLM_ARGS, MODEL
from flood_strategies import all_strategies, render_amp_rgb, render_coh_change_rgb
from phase1.llm_client import call_vision, extract_json

event = '20210727_Weihui'
PATCHES = f'D:/DisasterM3/UrbanSARFloods/testing_case_256/{event}'
GT_PATH = f'D:/DisasterM3/UrbanSARFloods/{event}/testing_case_orig/{event}/{event}_GT.tif'


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


with rasterio.open(GT_PATH) as ds:
    gt_full = ds.read(1)
files = sorted(glob.glob(os.path.join(PATCHES, '*_SAR.tif')))
vp = load_prompt('verifier_prompt.txt')
tmp = tempfile.mkdtemp()

rows = []  # (iou, flood_visible, mask_quality)
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
    # GE config F 掩膜 + 真实 IoU
    res = process_tile(sar, g, 'F', tmp, samples=1, no_vlm=True)
    pred = res['fused']
    gb = g > 0
    tp = (pred & gb).sum(); fp = (pred & ~gb).sum(); fn = (~pred & gb).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0
    # Qwen verifier
    amp_png = _save_png(render_amp_rgb(sar), os.path.join(tmp, 'a.png'))
    coh_png = _save_png(render_coh_change_rgb(sar), os.path.join(tmp, 'c.png'))
    ov_png = _save_png(_mask_overlay(render_amp_rgb(sar), pred), os.path.join(tmp, 'o.png'))
    try:
        text = call_vision(MODEL, vp, [amp_png, coh_png, ov_png], max_tokens=512,
                           timeout=180, **LLM_ARGS)
        v = extract_json(text)
        fv = v.get('flood_visible', True)
        mq = v.get('mask_quality', '?')
    except Exception:
        fv, mq = True, '?'
    rows.append((iou, fv, mq, tp, fp, fn))

print(f'Weihui patch={len(rows)}')
# 统计 flood_visible 与 IoU 关系
fv_true = [r[0] for r in rows if r[1] is True]
fv_false = [r[0] for r in rows if r[1] is False]
print(f'flood_visible=True: n={len(fv_true)} 平均IoU={np.mean(fv_true):.3f}')
print(f'flood_visible=False: n={len(fv_false)} 平均IoU={np.mean(fv_false):.3f}')

# 模拟语义门：flood_visible=False → 清空（tp=fp=0, fn 保留）
tp0 = fp0 = fn0 = 0
tp1 = fp1 = fn1 = 0
for iou, fv, mq, tp, fp, fn in rows:
    tp0 += tp; fp0 += fp; fn0 += fn
    if fv is True:
        tp1 += tp; fp1 += fp; fn1 += fn
    else:
        tp1 += 0; fp1 += 0; fn1 += fn  # 清空：tp/fp=0，fn 保留

def f1(t, f, n):
    p_ = t / (t + f) if t + f else 0
    r_ = t / (t + n) if t + n else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0

print(f'\n无语义门: F1={f1(tp0, fp0, fn0):.3f} P={tp0/(tp0+fp0):.3f} R={tp0/(tp0+fn0):.3f}')
print(f'语义门后: F1={f1(tp1, fp1, fn1):.3f} P={tp1/(tp1+fp1):.3f} R={tp1/(tp1+fn1):.3f}')
