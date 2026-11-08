"""图3 真实版：Weihui / Jubba_1 / Jubba_2 三行掩膜对比（SAR | GT | config F | CNN Unet++）
Jubba_2 的 patch 由脚本自动挑选（GT 洪水像素最多且 >=100 的 patch）。
输出：docs/01_论文/fig3_masks_3row.{png,pdf}
用法：python _render_fig3_masks3row.py
"""
import os, re, glob
import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp

os.environ['USF_NORM'] = 'zscore'
from data_loader import normalize
from flood_strategies import all_strategies, conn_filter

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = os.path.join(DATA, 'testing_case_256')
CONN_MIN, POST_THR = 500, -12.0
CKPT = os.path.join('ckpts_minmax', 'unet++_best_unetpp.pth')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None,
                         in_channels=8, classes=3).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()


def get_patch(sc, r, c):
    f = os.path.join(PATCHES, sc, f'{sc}_ID_{r}_{c}_SAR.tif')
    with rasterio.open(f) as ds:
        sar = ds.read().astype(np.float32)
    gtf = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
    with rasterio.open(gtf) as ds:
        gt = ds.read(1)[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
    return sar, gt


def config_f(sar):
    s = all_strategies(sar)
    f = s['int'] & s['chg']
    f = f & (np.nan_to_num(sar[7]) < POST_THR)
    return conn_filter(f, CONN_MIN)


def cnn_flood(sar):
    with torch.no_grad():
        p = model(torch.from_numpy(normalize(sar)).unsqueeze(0).to(device))
    return p.argmax(1).squeeze(0).cpu().numpy() > 0


def gray_bg(sar):
    vv = np.nan_to_num(sar[7], nan=-20.0)
    vh = np.nan_to_num(sar[6], nan=-20.0)
    a = 0.6 * vv + 0.4 * vh
    lo, hi = np.percentile(a, [2, 98])
    g = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.stack([g] * 3, axis=-1)


def overlay(bg, mask, color, alpha=0.65):
    out = bg.copy()
    for i in range(3):
        ch = out[..., i]
        ch[mask] = ch[mask] * (1 - alpha) + color[i] * alpha
    return out


def pick_jubba2_patch(sc='20231201_Jubba_2', min_px=100):
    gtf = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
    with rasterio.open(gtf) as ds:
        gtfull = ds.read(1)
    best, nvalid = None, 0
    for f in glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')):
        m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(f))
        r, c = int(m.group(1)), int(m.group(2))
        g = gtfull[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
        n = int((g > 0).sum())
        if n >= min_px:
            nvalid += 1
            if best is None or n > best[2]:
                best = (r, c, n)
    return best, nvalid


best, nvalid = pick_jubba2_patch()
print('Jubba_2 patches with >=100 flood px:', nvalid, '| picked (r,c,flood_px):', best)

SCENES = [('20210727_Weihui', 16, 35, 'Weihui  (r=16, c=35)'),
          ('20231201_Jubba_1', 22, 46, 'Jubba_1  (r=22, c=46)')]
if best is not None:
    SCENES.append(('20231201_Jubba_2', best[0], best[1],
                   f'Jubba_2  (r={best[0]}, c={best[1]})'))

fig, axes = plt.subplots(len(SCENES), 4, figsize=(7.16, 3.6 * len(SCENES) / 2))
for row, (sc, r, c, title) in enumerate(SCENES):
    sar, gt = get_patch(sc, r, c)
    bg = gray_bg(sar)
    cf = config_f(sar)
    cnn = cnn_flood(sar)
    print(f'{sc} ({r},{c}): GT flood={int((gt>0).sum())} | configF={int(cf.sum())} | CNN={int(cnn.sum())}')

    gtpanel = overlay(bg * 0.5, gt == 1, (0.05, 0.35, 0.95))
    gtpanel = overlay(gtpanel, gt == 2, (0.95, 0.12, 0.10))
    cfpanel = overlay(bg * 0.5, cf, (0.00, 0.75, 0.75))
    cnnpanel = overlay(bg * 0.5, cnn, (0.95, 0.80, 0.10))

    names = ['SAR (post VV/VH)', 'Ground truth', 'config F (zero-training)', 'CNN Unet++']
    for col, (im, name) in enumerate(zip([bg, gtpanel, cfpanel, cnnpanel], names)):
        ax = axes[row, col] if len(SCENES) > 1 else axes[col]
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(name if row == 0 else '', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(title, fontsize=8)

fig.legend(handles=[
    mpatches.Patch(facecolor=(0.05, 0.35, 0.95), label='GT open flood'),
    mpatches.Patch(facecolor=(0.95, 0.12, 0.10), label='GT urban flood'),
    mpatches.Patch(facecolor=(0.00, 0.75, 0.75), label='config F (rule)'),
    mpatches.Patch(facecolor=(0.95, 0.80, 0.10), label='CNN Unet++ (pred>0)'),
], loc='lower center', ncol=4, fontsize=7, frameon=False)
plt.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(DATA, 'docs', '01_论文', 'fig3_masks_3row')
fig.savefig(out + '.pdf', bbox_inches='tight')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
print('saved:', out)
