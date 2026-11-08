"""A1：逐事件城区洪水(FU, GT==2) F1 —— config F + 3 个 zscore CNN(seed42)。

协议（与 open-F1 对称，patch 级像素聚合，≥100 flood 像素过滤，Jubba 合并）：
  config F            : pred = 二值洪水掩膜
  CNN (pred>0 变体)    : pred = argmax > 0（任意洪水类 = 检测到洪水，与 open F1 同约定）
  CNN (pred==2 严格类) : pred = argmax == 2（城区类严格口径，config F 不适用）
城区 F1 = 像素聚合混淆矩阵：正类 GT==2 vs pred。
"""
import glob
import os
import re
import sys

_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.dirname(_REPRO_DIR)
sys.path.insert(0, _REPRO_DIR)
os.environ['USF_NORM'] = 'zscore'

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp
from data_loader import normalize
from flood_strategies import all_strategies, conn_filter

DATA = os.environ.get('USF_ROOT', _DATA_ROOT)
PATCHES = os.path.join(DATA, 'testing_case_256')
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
CONN_MIN, POST_THR = 500, -12.0

CKPTS = {
    'Unet':        os.path.join(_REPRO_DIR, 'ckpts_zscore', 'unet_best_unet_zscore.pth'),
    'Unet++':      os.path.join(_REPRO_DIR, 'ckpts_minmax', 'unet++_best_unetpp.pth'),
    'DeepLabV3+':  os.path.join(_REPRO_DIR, 'ckpts_minmax', 'deeplabv3+_best_deeplabv3p.pth'),
}
ARCH = {'Unet': 'unet', 'Unet++': 'unet++', 'DeepLabV3+': 'deeplabv3+'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

METHODS = ['configF', 'Unet', 'Unet++', 'DeepLabV3+']
# 每个方法输出两类掩膜：pred>0(flood 检测) 与 pred==2(严格城区类，仅 CNN)
VARIANTS = ['flood', 'class2']


def build_model(arch, ckpt):
    m = {'unet': smp.Unet, 'unet++': smp.UnetPlusPlus,
         'deeplabv3+': smp.DeepLabV3Plus}[arch](
        encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3).to(device)
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


def config_f_mask(sar):
    s = all_strategies(sar)
    f = s['int'] & s['chg']
    f = f & (np.nan_to_num(sar[7]) < POST_THR)
    return conn_filter(f, CONN_MIN)


cnn = {name: build_model(ARCH[name], p) for name, p in CKPTS.items()}

# acc[scene][method][gt_cls]['tp','fp','fn']，gt_cls: 1=open, 2=urban
from collections import defaultdict
acc = {sc: defaultdict(lambda: defaultdict(lambda: [0, 0, 0])) for sc in SCENES}

with torch.no_grad():
    for sc in SCENES:
        files = sorted(glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')))
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
            masks = {}
            masks['configF'] = {'flood': config_f_mask(sar)}
            for name, model in cnn.items():
                seg = model(torch.from_numpy(normalize(sar)).unsqueeze(0).to(device)
                            ).argmax(1).squeeze(0).cpu().numpy()
                masks[name] = {'flood': seg > 0, 'class2': seg == 2}
            for method, mv in masks.items():
                for gcls in (1, 2):
                    gbin = g == gcls
                    for var, pred in mv.items():
                        cm = acc[sc][method][f'{gcls}_{var}']
                        cm[0] += (pred & gbin).sum()
                        cm[1] += (pred & ~gbin).sum()
                        cm[2] += (~pred & gbin).sum()
        print(f'{sc}: {n_ev} patches', file=sys.stderr)


def show_block(title, var, gcls_label):
    labs = ['Weihui', 'Nova', 'Jubba_1', 'Jubba_2', 'Jubba合并', '3事件均值']
    print(f'\n== {title} ==')
    print(f'{"method":12s} ' + '  '.join(f'{x:>9s}' for x in labs))
    for method in METHODS:
        cells = []
        jub = [0, 0, 0]
        for sc in SCENES:
            cm = acc[sc][method].get(f'{gcls_label}_{var}', [0, 0, 0])
            cells.append(f1(*cm))
            if 'Jubba' in sc:
                for i in range(3):
                    jub[i] += cm[i]
        fj = f1(*jub)
        cells.append(fj)
        w = acc['20210727_Weihui'][method].get(f'{gcls_label}_{var}', [0, 0, 0])
        n = acc['20230609_NovaKakhovka'][method].get(f'{gcls_label}_{var}', [0, 0, 0])
        cells.append(np.mean([f1(*w), f1(*n), fj]))
        print(f'{method:12s} ' + '  '.join(f'{v:.3f}'.rjust(9) for v in cells))


for var, t in [('flood', '城区 F1：GT==2 vs 洪水检测 (pred>0/掩膜)'),
               ('class2', '城区 F1：GT==2 vs CNN 严格城区类 (pred==2，configF 不适用)')]:
    show_block(t, var, 2)
print('\n参考 CNN 城区 F1 文献值（整景/混合协议，勿直接引用）：Jubba_1 0.003 / Weihui 0.012 / Nova 0.000 / Jubba_2 0.050')
