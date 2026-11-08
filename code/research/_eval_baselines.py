"""通用 CNN 基线评测：任意 SMP 架构 + checkpoint → 同协议 open F1（与 config F 对比）。

用法：python _eval_baselines.py <arch> <ckpt_path>
arch ∈ {unet, unet++, fpn, pspnet, manet, deeplabv3+}
输出：每事件 open F1 + Jubba 合并 + 3 事件均值
"""
import glob
import os
import re
import sys

os.environ['USF_NORM'] = 'zscore'  # unet++/deeplabv3+ 训练用默认 zscore（unet 那批才是 minmax，用 _eval_multiseed）

# 路径从脚本位置推导：本脚本位于 <repo>/repro/，父目录即数据根
_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.dirname(_REPRO_DIR)
sys.path.insert(0, _REPRO_DIR)

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp
from data_loader import normalize

DATA = os.environ.get('USF_ROOT', _DATA_ROOT)
PATCHES = os.path.join(DATA, 'testing_case_256')
EVENTS = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ARCHES = {
    'unet': lambda: smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
    'unet++': lambda: smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
    'fpn': lambda: smp.FPN(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
    'pspnet': lambda: smp.PSPNet(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
    'manet': lambda: smp.MAnet(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
    'deeplabv3+': lambda: smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3),
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
                # 必须与训练一致地归一化（训练见 data_loader.normalize）；否则原始 dB 值
                # + NaN→0 喂给模型导致分布错位 → 全水/全背景假象（曾致 unet++ 误判为 0.124）
                inp = torch.from_numpy(normalize(sar)).unsqueeze(0).to(device)
                pred = model(inp).argmax(1).squeeze(0).cpu().numpy()
            pred = pred > 0  # 任意洪水类
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
    print(f'{arch}: Weihui={per["20210727_Weihui"]:.3f} Nova={per["20230609_NovaKakhovka"]:.3f} '
          f'Jubba_1={per["20231201_Jubba_1"]:.3f} Jubba_2={per["20231201_Jubba_2"]:.3f} '
          f'Jubba合并={jubba:.3f} 均值={mean3:.3f}')
