"""统一评测：4 测试事件 × testing_case_256 patch × 类别分解。
对比：复现 CNN / GE config A / GE config F(int∩chg) / GE config F+urban。
"""
import glob
import os
import re
import sys

sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

# 必须用 min-max 归一化（ckpts_minmax 用 min-max 训练；默认 z-score 会崩）
os.environ['USF_NORM'] = 'minmax'

import numpy as np
import rasterio
import torch
from data_loader import normalize
import segmentation_models_pytorch as smp
from flood_strategies import all_strategies, conn_filter

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256'
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']

device = 'cuda'
model = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                 in_channels=8, classes=3).to(device)
model.load_state_dict(torch.load('D:/DisasterM3/UrbanSARFloods/repro/ckpts_minmax/unet_best.pth',
                                 map_location=device))
model.eval()


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


@torch.no_grad()
def cnn_pred(sar):
    inp = torch.from_numpy(normalize(sar)).unsqueeze(0).to(device)
    return model(inp).argmax(1).squeeze(0).cpu().numpy()


def ge_masks(sar, conn_min=500, post_thr=-12.0):
    s = all_strategies(sar)
    f_base = s['int'] & s['chg']
    f_base = f_base & (np.nan_to_num(sar[7]) < post_thr)  # 暗水约束
    return {
        'A': s['int'],
        'F': conn_filter(f_base, conn_min),
        'F+urban': conn_filter(f_base | s['urban'], conn_min),
    }


print('事件, 方法, openF1, urbanF1, allF1')
for ev in EVENTS:
    patch_dir = os.path.join(PATCHES, ev)
    files = sorted(glob.glob(os.path.join(patch_dir, '*_SAR.tif')))
    if not files:
        print(f'{ev}: 无 patches，跳过')
        continue
    gt_path = os.path.join(DATA, ev, 'testing_case_orig', ev, f'{ev}_GT.tif')
    if not os.path.exists(gt_path):
        print(f'{ev}: 无 GT，跳过')
        continue
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)

    acc = {m: {'open': [0, 0, 0], 'urban': [0, 0, 0], 'all': [0, 0, 0],
               'open_tile': [0, 0, 0], 'urban_tile': [0, 0, 0]}
           for m in ['CNN', 'A', 'F', 'F+urban']}
    n_patch = 0
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
        n_patch += 1
        preds = {'CNN': cnn_pred(sar) > 0}
        preds.update(ge_masks(sar))
        has_open = (g == 1).sum() >= 100
        has_urban = (g == 2).sum() >= 100
        for m, pred in preds.items():
            for cls, gbin in (('open', g == 1), ('urban', g == 2), ('all', g > 0)):
                acc[m][cls][0] += (pred & gbin).sum()
                acc[m][cls][1] += (pred & ~gbin).sum()
                acc[m][cls][2] += (~pred & gbin).sum()
            if has_open:
                acc[m]['open_tile'][0] += (pred & (g == 1)).sum()
                acc[m]['open_tile'][1] += (pred & (g != 1)).sum()
                acc[m]['open_tile'][2] += (~pred & (g == 1)).sum()
            if has_urban:
                acc[m]['urban_tile'][0] += (pred & (g == 2)).sum()
                acc[m]['urban_tile'][1] += (pred & (g != 2)).sum()
                acc[m]['urban_tile'][2] += (~pred & (g == 2)).sum()

    for m in ['CNN', 'A', 'F', 'F+urban']:
        row = [ev, m]
        for cls in ['open', 'urban', 'all', 'open_tile', 'urban_tile']:
            tp, fp, fn = acc[m][cls]
            p_ = tp / (tp + fp) if tp + fp else 0
            r_ = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
            row.append(f'{f1:.3f}')
        print(', '.join(row), f'  (patch={n_patch})')
