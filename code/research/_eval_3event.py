"""3 事件聚合口径：Jubba 合并为单事件（对标 BCP / 原论文）。

输出两种视图：
  View A（4 场景）：Weihui, NovaKakhovka, Jubba_1, Jubba_2 各自独立
  View B（3 事件）：Weihui, NovaKakhovka, Jubba(=Jubba_1+Jubba_2 混淆矩阵相加)

配置：CNN / GE-A / GE-C(确定性) / GE-F(确定性, conn=500)，open 与 all 两个 F1。
"""
import glob
import os
import re
import sys

# 路径从脚本位置推导：本脚本位于 <repo>/repro/，父目录即数据根
_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.dirname(_REPRO_DIR)
sys.path.insert(0, _REPRO_DIR)
os.environ['USF_NORM'] = 'minmax'

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
CONN_MIN = 500
POST_THR = -12.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# CNN 列为可选：若未提供 checkpoint，则只输出 config A/C/F（config F 是论文主结果，不依赖 checkpoint）。
_CNN_CKPT = os.path.join(_REPRO_DIR, 'ckpts_minmax', 'unet_best.pth')
model = None
if os.path.exists(_CNN_CKPT):
    model = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                     in_channels=8, classes=3).to(device)
    model.load_state_dict(torch.load(_CNN_CKPT, map_location=device))
    model.eval()
else:
    print('[提示] 未找到 CNN checkpoint，跳过 CNN 列（config A/C/F 照常输出）')
CONFIG_NAMES = (['CNN'] if model is not None else []) + ['A', 'C', 'F']


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


@torch.no_grad()
def cnn_pred(sar):
    if model is None:
        return None
    inp = torch.from_numpy(normalize(sar)).unsqueeze(0).to(device)
    return model(inp).argmax(1).squeeze(0).cpu().numpy()


def masks(sar):
    s = all_strategies(sar)
    a = s['int']
    scene = classify_scene(sar)
    labels = {"scene": scene.get("scene"), "scene_density": scene.get("scene_density")}
    top = route(labels)["expectedBest"] or "int"
    c = s[top]
    f = s['int'] & s['chg']
    f = f & (np.nan_to_num(sar[7]) < POST_THR)
    f = conn_filter(f, CONN_MIN)
    out = {'A': a, 'C': c, 'F': f}
    p = cnn_pred(sar)
    if p is not None:
        out['CNN'] = p > 0
    return out


def f1(tp, fp, fn):
    p_ = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0


# 每场景每配置的混淆矩阵
acc = {}  # scene -> config -> {'open': [tp,fp,fn], 'all': [tp,fp,fn]}
for sc in SCENES:
    patch_dir = os.path.join(PATCHES, sc)
    files = sorted(glob.glob(os.path.join(patch_dir, '*_SAR.tif')))
    gt_path = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
    if not (files and os.path.exists(gt_path)):
        continue
    with rasterio.open(gt_path) as ds:
        gt_full = ds.read(1)
    acc[sc] = {m: {'open': [0, 0, 0], 'all': [0, 0, 0]} for m in CONFIG_NAMES}
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
        preds = masks(sar)
        for m, pred in preds.items():
            acc[sc][m]['open'][0] += (pred & (g == 1)).sum()
            acc[sc][m]['open'][1] += (pred & (g != 1)).sum()
            acc[sc][m]['open'][2] += (~pred & (g == 1)).sum()
            acc[sc][m]['all'][0] += (pred & (g > 0)).sum()
            acc[sc][m]['all'][1] += (pred & (g == 0)).sum()
            acc[sc][m]['all'][2] += (~pred & (g > 0)).sum()

CONFIGS = CONFIG_NAMES


def show_row(name, cm, note=''):
    cells = [f'{f1(*cm["open"]):.3f}', f'{f1(*cm["all"]):.3f}']
    print(f'  {name:16s} open={cells[0]}  all={cells[1]}  {note}')


print('=' * 70)
print('视图 A · 4 场景独立')
print('=' * 70)
for sc in SCENES:
    print(f'-- {sc} --')
    for m in CONFIGS:
        show_row(m, acc[sc][m])

print()
print('=' * 70)
print('视图 B · 3 事件（Jubba 合并 = Jubba_1 + Jubba_2）')
print('=' * 70)
# Jubba 合并
jubba = {m: {'open': [0, 0, 0], 'all': [0, 0, 0]} for m in CONFIGS}
for m in CONFIGS:
    for k in ('open', 'all'):
        for i in range(3):
            jubba[m][k][i] = acc['20231201_Jubba_1'][m][k][i] + acc['20231201_Jubba_2'][m][k][i]

for name, sc in [('Weihui', '20210727_Weihui'), ('NovaKakhovka', '20230609_NovaKakhovka')]:
    print(f'-- {name} --')
    for m in CONFIGS:
        show_row(m, acc[sc][m])
print('-- Jubba (合并) --')
for m in CONFIGS:
    show_row(m, jubba[m])

print()
print('均值（3 事件，open F1 / all F1）:')
for m in CONFIGS:
    evs = [acc['20210727_Weihui'][m], acc['20230609_NovaKakhovka'][m], jubba[m]]
    print(f'  {m:16s} open={np.mean([f1(*e["open"]) for e in evs]):.3f}  '
          f'all={np.mean([f1(*e["all"]) for e in evs]):.3f}')
print()
print('对照 BCP（3 事件，all-F1 约）：Weihui 0.75, Jubba 0.68, Nova 0.49')
