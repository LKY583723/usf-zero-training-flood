"""评测 Jubba 的 testing_case_256 patches（论文口径：独立 patch + 像素聚合）。
用法：python _eval_jubba_patches.py 20231201_Jubba_1|20231201_Jubba_2
"""
import glob
import os
import re
import sys
sys.path.insert(0, '.')
import numpy as np
import rasterio
import torch
from data_loader import normalize
import segmentation_models_pytorch as smp

ev = sys.argv[1]
device = 'cuda'
model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3).to(device)
model.load_state_dict(torch.load('D:/DisasterM3/UrbanSARFloods/repro/ckpts_minmax/unet_best.pth',
                                 map_location=device))
model.eval()

PATCH_DIR = f'D:/DisasterM3/UrbanSARFloods/testing_case_256/{ev}'
GT_PATH = f'D:/DisasterM3/UrbanSARFloods/{ev}/testing_case_orig/{ev}/{ev}_GT.tif'

with rasterio.open(GT_PATH) as ds:
    gt_full = ds.read(1)
print('整景 GT:', gt_full.shape, flush=True)

files = sorted(glob.glob(os.path.join(PATCH_DIR, '*_SAR.tif')))
print('patch 文件:', len(files), flush=True)

def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))

@torch.no_grad()
def infer(sar_path):
    with rasterio.open(sar_path) as ds:
        sar = ds.read().astype(np.float32)
    sar = normalize(sar)
    inp = torch.from_numpy(sar).unsqueeze(0).to(device)
    return model(inp).argmax(1).squeeze(0).cpu().numpy()

# 像素聚合
acc = {'open': [0,0,0], 'urban': [0,0,0], 'all': [0,0,0]}
n_flood = 0
for f in files:
    r, c = parse_id(f)
    y0, x0 = r*256, c*256
    if y0+256 > gt_full.shape[0] or x0+256 > gt_full.shape[1]:
        continue
    g = gt_full[y0:y0+256, x0:x0+256]
    if (g > 0).sum() < 100:
        continue
    p = infer(f)
    n_flood += 1
    pred = p > 0
    for cls, gbin in (('open', g==1), ('urban', g==2), ('all', g>0)):
        acc[cls][0] += (pred & gbin).sum()
        acc[cls][1] += (pred & ~gbin).sum()
        acc[cls][2] += (~pred & gbin).sum()

print(f'洪水 patch: {n_flood}', flush=True)
for cls, (tp, fp, fn) in acc.items():
    p_ = tp/(tp+fp) if tp+fp else 0
    r_ = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p_*r_/(p_+r_) if p_+r_ else 0
    print(f'{cls}: F1={f1:.3f} P={p_:.3f} R={r_:.3f} (tp={tp}, fp={fp}, fn={fn})', flush=True)
