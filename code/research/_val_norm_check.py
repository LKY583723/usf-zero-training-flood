"""官方验证集复现：确认训练归一化方式（哪个 norm 能复现训练日志的 val IoU）。

用法：python _val_norm_check.py <arch> <ckpt> <norm> <ref_open_iou>
"""
import os
import sys

norm = sys.argv[3]
os.environ['USF_NORM'] = norm
sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')

import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from data_loader import UrbanSARFloodDataset

arch, ckpt = sys.argv[1], sys.argv[2]
ref = float(sys.argv[4])

MODELS = {
    'unet': lambda: smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
    'unet++': lambda: smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
    'deeplabv3+': lambda: smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
}

model = MODELS[arch]().to('cuda')
model.load_state_dict(torch.load(ckpt, map_location='cuda'))
model.eval()

ds = UrbanSARFloodDataset('D:/DisasterM3/UrbanSARFloods/urban_sar_floods', 'valid', 256,
                          augment=False, preload=True)
dl = DataLoader(ds, batch_size=12, shuffle=False, num_workers=0)

tp = np.zeros(3); fp = np.zeros(3); fn = np.zeros(3)
with torch.no_grad():
    for sar, gt in dl:
        pred = model(sar.to('cuda')).argmax(1).cpu().numpy()
        gt = gt.numpy()
        for c in range(3):
            p = pred == c; t = gt == c
            tp[c] += (p & t).sum(); fp[c] += (p & ~t).sum(); fn[c] += (~p & t).sum()
ious = tp / np.maximum(tp + fp + fn, 1)
print(f'[{norm}] {arch} val open IoU = {ious[1]:.4f} (训练日志参考 = {ref:.4f})')
