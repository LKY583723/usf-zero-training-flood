"""城区洪水 F1 正式表（采纳 2026-09-05 口径决定）。

主口径（detection semantics，与主表 open F1 完全同构）：
  正类 = GT==2（城区洪水像素）；预测正类 = 方法的洪水掩膜（config A/C/F 二值；CNN pred>0）
  逐场景像素聚合；同一份 patch 名单 + GT 洪水≥100px 过滤；Jubba 合并 = 混淆矩阵相加。
补充口径（脚注）：CNN class-matching FU F1 = pred==2 vs GT==2；GE 无类输出标 "—"。
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
from scene_classifier import classify_scene
from flood_router import route

DATA = os.environ.get('USF_ROOT', _DATA_ROOT)
PATCHES = os.path.join(DATA, 'testing_case_256')
SCENES = ['20210727_Weihui', '20230609_NovaKakhovka',
          '20231201_Jubba_1', '20231201_Jubba_2']
CONN_MIN, POST_THR = 500, -12.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CNN_CKPTS = {
    'Unet':       ('unet',        os.path.join(_REPRO_DIR, 'ckpts_zscore', 'unet_best_unet_zscore.pth')),
    'Unet++':     ('unet++',      os.path.join(_REPRO_DIR, 'ckpts_minmax', 'unet++_best_unetpp.pth')),
    'DeepLabV3+': ('deeplabv3+',  os.path.join(_REPRO_DIR, 'ckpts_minmax', 'deeplabv3+_best_deeplabv3p.pth')),
}
RULES = ['A', 'C', 'F']
METHODS = RULES + list(CNN_CKPTS)


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


def rule_masks(sar):
    s = all_strategies(sar)
    a = s['int']
    labels = {'scene': classify_scene(sar).get('scene'),
              'scene_density': classify_scene(sar).get('scene_density')}
    top = route(labels)['expectedBest'] or 'int'
    c = s[top]
    f = s['int'] & s['chg']
    f = f & (np.nan_to_num(sar[7]) < POST_THR)
    f = conn_filter(f, CONN_MIN)
    return {'A': a, 'C': c, 'F': f}


cnn = {name: build_model(arch, ck) for name, (arch, ck) in CNN_CKPTS.items()}

# acc[scene][method]['urban_flood'|'urban_class2'][tp,fp,fn]
acc = {sc: {m: {'uf': [0, 0, 0], 'c2': [0, 0, 0]} for m in METHODS} for sc in SCENES}

with torch.no_grad():
    for sc in SCENES:
        files = sorted(glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')))
        gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
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
            masks = rule_masks(sar)
            for name, model in cnn.items():
                seg = model(torch.from_numpy(normalize(sar)).unsqueeze(0).to(device)
                            ).argmax(1).squeeze(0).cpu().numpy()
                masks[name] = {'flood': seg > 0, 'class2': seg == 2}
            g2 = g == 2
            for m in METHODS:
                msk = masks[m]
                if isinstance(msk, dict):  # CNN
                    cm = acc[sc][m]['uf']
                    pred = msk['flood']
                else:
                    cm = acc[sc][m]['uf']
                    pred = msk
                cm[0] += (pred & g2).sum(); cm[1] += (pred & ~g2).sum(); cm[2] += (~pred & g2).sum()
                if isinstance(msk, dict):
                    cm = acc[sc][m]['c2']
                    pred2 = msk['class2']
                    cm[0] += (pred2 & g2).sum(); cm[1] += (pred2 & ~g2).sum(); cm[2] += (~pred2 & g2).sum()
        print(f'{sc}: {n_ev} patches', file=sys.stderr)

LONG = {'20210727_Weihui': 'Weihui', '20230609_NovaKakhovka': 'Nova',
        '20231201_Jubba_1': 'Jubba_1', '20231201_Jubba_2': 'Jubba_2'}


def emit(title, key, methods, na=None):
    print(f'\n== {title} ==')
    hdr = 'method'.ljust(11) + ''.join(x.rjust(9) for x in
                                       ['Weihui', 'Nova', 'Jubba_1', 'Jubba_2', 'Jub合并', '3ev均值'])
    print(hdr)
    for m in methods:
        if na and m in na:
            print(m.ljust(11) + ''.join('—'.rjust(9) for _ in range(6)))
            continue
        cells = []
        jub = [0, 0, 0]
        for sc in SCENES:
            cm = acc[sc][m][key]
            cells.append(f1(*cm))
            if 'Jubba' in sc:
                for i in range(3):
                    jub[i] += cm[i]
        fj = f1(*jub)
        cells.append(fj)
        w = acc['20210727_Weihui'][m][key]; n = acc['20230609_NovaKakhovka'][m][key]
        cells.append(np.mean([f1(*w), f1(*n), fj]))
        row = m.ljust(11) + ''.join(f'{v:.3f}'.rjust(9) for v in cells)
        print(row)


emit('主表 · 城区洪水 F1（GT==2 vs 洪水检测 pred>0/掩膜）', 'uf', METHODS)
emit('脚注 · CNN class-matching FU F1（pred==2 vs GT==2；GE 无类输出=—）', 'c2',
     METHODS, na=('A', 'C', 'F'))
