"""3-seed unet++ 集成评测：平均 softmax 概率 → argmax。

用途：检验"多 seed 集成"这条合法增强路线能否超过 config F(0.654)。
同时逐 seed 复算 open F1，验证脚本正确性：
  seed42→0.555, seed0→0.489, seed123→0.511 (mean 0.518±0.034)
口径与 _eval_3event.py 完全一致：≥100 flood 像素过滤、像素聚合、Jubba 合并。
"""
import glob
import os
import re
import sys

_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.dirname(_REPRO_DIR)
sys.path.insert(0, _REPRO_DIR)
os.environ['USF_NORM'] = 'zscore'  # 三个 checkpoint 均为 zscore 训练（val 复现验证过）

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp
from data_loader import normalize

DATA = os.environ.get('USF_ROOT', _DATA_ROOT)
PATCHES = os.path.join(DATA, 'testing_case_256')
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']

CKPTS = [
    ('seed42',  os.path.join(_REPRO_DIR, 'ckpts_minmax',  'unet++_best_unetpp.pth')),
    ('seed0',   os.path.join(_REPRO_DIR, 'ckpts_zscore',  'unet++_best_unetpp_seed0.pth')),
    ('seed123', os.path.join(_REPRO_DIR, 'ckpts_zscore',  'unet++_best_unetpp_seed123.pth')),
]
LABELS = [name for name, _ in CKPTS] + ['ensemble']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(ckpt):
    m = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None,
                         in_channels=8, classes=3).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


models = {name: build_model(ck) for name, ck in CKPTS}

# scene -> label -> {'open':[tp,fp,fn], 'all':[tp,fp,fn]}
acc = {sc: {lb: {'open': [0, 0, 0], 'all': [0, 0, 0]} for lb in LABELS} for sc in SCENES}

with torch.no_grad():
    for sc in SCENES:
        patch_dir = os.path.join(PATCHES, sc)
        files = sorted(glob.glob(os.path.join(patch_dir, '*_SAR.tif')))
        gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
        if not (files and os.path.exists(gt_path)):
            continue
        with rasterio.open(gt_path) as ds:
            gt_full = ds.read(1)
        n_ev = 0
        for f in files:
            r, c = parse_id(f)
            y0, x0 = r * 256, c * 256
            if y0 + 256 > gt_full.shape[0] or x0 + 256 > gt_full.shape[1]:
                continue
            g = gt_full[y0:y0 + 256, x0:x0 + 256]
            if (g > 0).sum() < 100:
                continue
            n_ev += 1
            with rasterio.open(f) as ds:
                sar = ds.read().astype(np.float32)
            inp = torch.from_numpy(normalize(sar)).unsqueeze(0).to(device)
            probs = {}
            for name, model in models.items():
                probs[name] = torch.softmax(model(inp), 1)  # (1,3,256,256)
            probs['ensemble'] = torch.mean(torch.stack(
                [probs[n] for n, _ in CKPTS], dim=0), dim=0)
            for lb, pr in probs.items():
                pred = pr.argmax(1).squeeze(0).cpu().numpy() > 0
                cm = acc[sc][lb]
                cm['open'][0] += (pred & (g == 1)).sum()
                cm['open'][1] += (pred & (g != 1)).sum()
                cm['open'][2] += (~pred & (g == 1)).sum()
                cm['all'][0] += (pred & (g > 0)).sum()
                cm['all'][1] += (pred & (g == 0)).sum()
                cm['all'][2] += (~pred & (g > 0)).sum()
        print(f'{sc}: evaluated {n_ev} flood patches', file=sys.stderr)

print('=' * 62)
print('4 场景独立 open F1')
print('=' * 62)
for sc in SCENES:
    row = '  '.join(f'{lb}={f1(*acc[sc][lb]["open"]):.3f}' for lb in LABELS)
    print(f'{sc}\n  {row}')

# Jubba 合并 + 3 事件均值
jubba = {lb: {'open': [0, 0, 0], 'all': [0, 0, 0]} for lb in LABELS}
for lb in LABELS:
    for k in ('open', 'all'):
        for i in range(3):
            jubba[lb][k][i] = (acc['20231201_Jubba_1'][lb][k][i]
                               + acc['20231201_Jubba_2'][lb][k][i])

print()
print('=' * 62)
print('3 事件视图（Jubba 合并）· open F1 / all F1')
print('=' * 62)
for lb in LABELS:
    w, n, j = acc['20210727_Weihui'][lb], acc['20230609_NovaKakhovka'][lb], jubba[lb]
    print(f'{lb:9s}  Weihui={f1(*w["open"]):.3f}  Nova={f1(*n["open"]):.3f}  '
          f'Jubba={f1(*j["open"]):.3f}  mean_open={np.mean([f1(*w["open"]), f1(*n["open"]), f1(*j["open"])]):.3f}  '
          f'mean_all={np.mean([f1(*w["all"]), f1(*n["all"]), f1(*j["all"])]):.3f}')

print()
print('参考：config F = Weihui 0.773, Nova 0.367, Jubba 0.822, mean 0.654')
