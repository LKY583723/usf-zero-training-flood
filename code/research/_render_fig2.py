"""A3 图2：config F vs CNN(Unet++) vs GT 掩膜对比（Weihui + Jubba_1）。

每场景一行：SAR(灰度, 水暗陆亮) | GT | config F(青) | CNN Unet++(黄)
三列(GT/configF/CNN)共用同一灰度背景 + 彩色覆盖；右下图例。
输出：UrbanSARFloods/docs/01_论文/fig2_mask_comparison.png
"""
import os, sys, re
import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import segmentation_models_pytorch as smp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def parse_id(fn):
    m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(fn))
    return int(m.group(1)), int(m.group(2))


def get_patch(sc, r, c):
    pd = os.path.join(PATCHES, sc)
    f = os.path.join(pd, f'{sc}_ID_{r}_{c}_SAR.tif')
    with rasterio.open(f) as ds:
        sar = ds.read().astype(np.float32)
    gtf = os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')
    with rasterio.open(gtf) as ds:
        gt = ds.read(1)[r*256:(r+1)*256, c*256:(c+1)*256]
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
    """灰度背景：水暗、陆亮（高后向散射 = 亮）。"""
    vv = np.nan_to_num(sar[7], nan=-20.0)
    vh = np.nan_to_num(sar[6], nan=-20.0)
    m = (vv + vh) / 2.0                 # dB，水约 −20~−15，陆约 −8~−2
    g = np.clip((m + 18.0) / 16.0, 0, 1)  # −18 dB→0(暗水)  −2 dB→1(亮陆)
    return np.stack([g] * 3, axis=-1)


def overlay(bg, mask, color, alpha=0.65):
    out = bg.copy()
    for i in range(3):
        ch = out[..., i]
        ch[mask] = ch[mask] * (1 - alpha) + color[i] * alpha
    return out


SCENES = [('20210727_Weihui', 16, 35, 'Weihui  (r=16, c=35)'),
          ('20231201_Jubba_1', 22, 46, 'Jubba_1  (r=22, c=46)')]

fig, axes = plt.subplots(2, 4, figsize=(15, 8.8))
for row, (sc, r, c, title) in enumerate(SCENES):
    sar, gt = get_patch(sc, r, c)
    bg = gray_bg(sar)
    cf = config_f(sar)
    cnn = cnn_flood(sar)

    # 三列(GT/configF/CNN)共用同一灰度背景，只是覆盖色不同
    gtpanel = overlay(bg, gt == 1, (0.05, 0.4, 1.0), alpha=0.7)
    gtpanel = overlay(gtpanel, gt == 2, (1.0, 0.15, 0.1), alpha=0.85)
    cfpanel = overlay(bg, cf, (0.0, 0.75, 0.75), alpha=0.65)
    cnnpanel = overlay(bg, cnn, (0.95, 0.8, 0.1), alpha=0.65)

    panels = [bg, gtpanel, cfpanel, cnnpanel]
    for col, (im, name) in enumerate(zip(
            panels,
            ['SAR (post VV/VH)', 'Ground truth', 'config F (zero-training)', 'CNN Unet++'])):
        ax = axes[row, col]
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(name if row == 0 else '', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(title, fontsize=11)

handles = [Patch(color=(0.05, 0.4, 1.0), label='open flood (GT=1)'),
           Patch(color=(1.0, 0.15, 0.1), label='urban flood (GT=2)'),
           Patch(color=(0.0, 0.75, 0.75), label='config F'),
           Patch(color=(0.95, 0.8, 0.1), label='CNN Unet++ (pred>0)')]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=11,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Flood masks on UrbanSARFloods test patches (gray = SAR backscatter, water dark)',
             fontsize=13)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out = os.path.join(DATA, 'docs', '01_论文', 'fig2_mask_comparison.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print('saved:', out)
