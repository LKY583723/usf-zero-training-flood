"""多 seed CNN 评测（修正协议）：预测任意洪水类(pred>0) vs GT open(g==1)。

与 _eval_unified.py 报告口径一致。仅评测完整训练完成的 seed。
"""
import glob
import os
import re
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')
os.environ['USF_NORM'] = 'minmax'

import numpy as np
import rasterio
import torch
from data_loader import normalize
import segmentation_models_pytorch as smp

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256'
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
CKPT_DIR = 'D:/DisasterM3/UrbanSARFloods/repro/ckpts_minmax'
SEEDS = {'seed42': 'unet_best.pth',          # 现有报告 checkpoint（完整）
         'seed0': 'unet_best_seed0.pth',     # 完整 100 轮（塌缩：测试全背景）
         'seed1': 'unet_best_seed1.pth',
         'seed2': 'unet_best_seed2.pth',
         'seed3': 'unet_best_seed3.pth'}
device = 'cuda'


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


@torch.no_grad()
def cnn_pred(model, sar):
    inp = torch.from_numpy(normalize(sar)).unsqueeze(0).to(device)
    return model(inp).argmax(1).squeeze(0).cpu().numpy()


def eval_seed(ckpt):
    model = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                     in_channels=8, classes=3).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    acc = {ev: [0, 0, 0] for ev in EVENTS}
    for ev in EVENTS:
        files = sorted(glob.glob(os.path.join(PATCHES, ev, '*_SAR.tif')))
        gt_path = os.path.join(DATA, ev, 'testing_case_orig', ev, f'{ev}_GT.tif')
        with rasterio.open(gt_path) as ds:
            gt_full = ds.read(1)
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
            pred = cnn_pred(model, sar) > 0  # 预测任意洪水类（对齐报告口径）
            a = acc[ev]
            a[0] += (pred & (g == 1)).sum()
            a[1] += (pred & (g != 1)).sum()
            a[2] += (~pred & (g == 1)).sum()
    j = [0, 0, 0]
    for i in range(3):
        j[i] = acc['20231201_Jubba_1'][i] + acc['20231201_Jubba_2'][i]
    per_scene = {ev: f1(*acc[ev]) for ev in EVENTS}
    jubba = f1(*j)
    mean3 = np.mean([per_scene['20210727_Weihui'], per_scene['20230609_NovaKakhovka'], jubba])
    return per_scene, jubba, mean3


print('seed, Weihui, NovaKakhovka, Jubba_1, Jubba_2, Jubba(合并), 3事件均值')
rows = {}
for name, ckpt in SEEDS.items():
    path = os.path.join(CKPT_DIR, ckpt)
    if not os.path.exists(path):
        print(f'{name}: 缺失 {path}，跳过')
        continue
    per_scene, jubba, mean3 = eval_seed(path)
    rows[name] = (per_scene, jubba, mean3)
    print(f'{name}, ' + ', '.join(f'{per_scene[ev]:.3f}' for ev in EVENTS)
          + f', {jubba:.3f}, {mean3:.3f}')

if len(rows) >= 2:
    print('\n=== 跨 seed 统计（3 事件均值）===')
    means = np.array([r[2] for r in rows.values()])
    print(f'n={len(rows)}  mean={means.mean():.3f}  std={means.std():.3f}  min={means.min():.3f}  max={means.max():.3f}')
    print('\n=== 每事件跨 seed ===')
    for ev in EVENTS:
        vals = np.array([rows[k][0][ev] for k in rows])
        print(f'{ev}: {vals.mean():.3f} ± {vals.std():.3f}')
    jvals = np.array([rows[k][1] for k in rows])
    print(f'Jubba(合并): {jvals.mean():.3f} ± {jvals.std():.3f}')
