"""CNN 基线训练/评测归一化一致性诊断。

用法：python _diag_norm_fix.py <arch> <ckpt> [norm]
  norm ∈ {zscore, minmax}（默认 zscore，即训练实际使用的归一化）
输出：每事件 open F1 + Jubba 合并 + 3 事件均值
"""
import glob
import os
import re
import sys

# 必须在 import data_loader 前设置（NORM_MODE 在 import 时固定）
norm = sys.argv[3] if len(sys.argv) > 3 else 'zscore'
os.environ['USF_NORM'] = norm
sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp
from data_loader import normalize

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = 'D:/DisasterM3/UrbanSARFloods/testing_case_256'
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
device = 'cuda'

ARCHES = {
    'unet': lambda: smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
    'unet++': lambda: smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
}


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


def eval_baseline(arch, ckpt):
    model = ARCHES[arch]().to(device)
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
            with torch.no_grad():
                inp = torch.from_numpy(normalize(sar)).unsqueeze(0).to(device)
                pred = model(inp).argmax(1).squeeze(0).cpu().numpy()
            pred = pred > 0
            a = acc[ev]
            a[0] += (pred & (g == 1)).sum()
            a[1] += (pred & (g != 1)).sum()
            a[2] += (~pred & (g == 1)).sum()
    j = [0, 0, 0]
    for i in range(3):
        j[i] = acc['20231201_Jubba_1'][i] + acc['20231201_Jubba_2'][i]
    per = {ev: f1(*acc[ev]) for ev in EVENTS}
    jubba = f1(*j)
    mean3 = np.mean([per['20210727_Weihui'], per['20230609_NovaKakhovka'], jubba])
    return per, jubba, mean3


if __name__ == '__main__':
    arch, ckpt = sys.argv[1], sys.argv[2]
    per, jubba, mean3 = eval_baseline(arch, ckpt)
    print(f'[{norm}] {arch}: Weihui={per["20210727_Weihui"]:.3f} '
          f'Nova={per["20230609_NovaKakhovka"]:.3f} '
          f'Jubba_1={per["20231201_Jubba_1"]:.3f} '
          f'Jubba_2={per["20231201_Jubba_2"]:.3f} '
          f'Jubba合并={jubba:.3f} 均值={mean3:.3f}')
