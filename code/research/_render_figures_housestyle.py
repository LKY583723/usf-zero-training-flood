"""IEEE 合规 + figures4papers house style 绘图脚本（三张图）
- 画布宽固定 7.16 in（IEEE 双栏通栏），字号 ≥8 pt，线宽 ≥0.75 pt
- 矢量 PDF + 300 dpi PNG
- Fig.1 首页总览图：Inputs | config F 流水线 | Evaluation
  + 监督对照带 + 三条结论带（开阔水胜负 / 闸门决定性 / 物理边界）
输出（docs/01_论文/）：
  fig1_architecture_housestyle.{png,pdf}
  fig2_gate_flow_col.{png,pdf}
  fig3_masks_3row.{png,pdf}
用法：python _render_figures_housestyle.py
"""
import os, re, glob
import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Patch
import segmentation_models_pytorch as smp

os.environ['USF_NORM'] = 'zscore'
from data_loader import normalize
from flood_strategies import all_strategies, conn_filter

# ---------------- IEEE-compliant rcParams（正文 Times，图内文字同族） ----------------
matplotlib.rcParams.update({
    "font.family": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "mathtext.fontset": "stix",          # 数学符号与 Times 同族（STIX）
    "axes.linewidth": 1.0,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA", "blue_tint": "#E8F0F8",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342", "red_tint": "#FDEDEB",
    "neutral": "#CFCECE", "gray_tint": "#F2F2F2", "highlight": "#FFD700",
}
BLACK = "#1A1A1A"
GRAY_TXT = "#6E6E6E"

DATA = 'D:/DisasterM3/UrbanSARFloods'
PATCHES = os.path.join(DATA, 'testing_case_256')
OUT = os.path.join(DATA, 'docs', '01_论文')
CONN_MIN, POST_THR = 500, -12.0
CKPT = os.path.join('ckpts_minmax', 'unet++_best_unetpp.pth')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def box(ax, x, y, w, h, text, fc, ec=BLACK, fs=8, lw=0.9, ls='-', tc=BLACK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle='round,pad=0.012,rounding_size=0.05',
                 linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc,
            zorder=4, linespacing=1.35)


def zone(ax, x0, y0, x1, y1, fc, ec='#BBBBBB', ls='-'):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle='round,pad=0.02,rounding_size=0.08',
                 linewidth=0.8, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=1))


def badge(ax, x, y, n, label, fs=9, r=0.19):
    ax.add_patch(Circle((x, y), r, facecolor=PALETTE['blue_main'],
                        edgecolor='none', zorder=4))
    ax.text(x, y, str(n), ha='center', va='center', fontsize=7, color='white', zorder=5)
    ax.text(x + r * 1.7, y, label, ha='left', va='center', fontsize=fs, color=BLACK, zorder=5)


def arrow(ax, x1, y1, x2, y2, color=BLACK, lw=1.0, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=9, linewidth=lw, color=color, linestyle=ls, zorder=2))


def export(fig, name, pad=0.1):
    """保存 PNG+PDF；PDF 里的位图素材按 300 dpi 栅格化（IEEE 要求 ≥300 dpi），文字仍是矢量。"""
    used = name
    for ext, kw in (('png', dict(dpi=300)), ('pdf', dict(dpi=300))):
        path = os.path.join(OUT, f'{used}.{ext}')
        try:
            fig.savefig(path, bbox_inches='tight', pad_inches=pad, **kw)
        except PermissionError:
            used = name + '_v2'
            path = os.path.join(OUT, f'{used}.{ext}')
            fig.savefig(path, bbox_inches='tight', **kw)
            print('  (原文件被占用，改存为)', used)
    print('saved:', used)


# ================= Fig 1 =================
def fig_architecture():
    """首页总览图：输入 → 零训练规则流水线（config F）→ 评测；底部为监督对照带 + 三条结论带。"""
    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 10.0); ax.axis('off')

    # 主区由旧坐标 (1.60–6.30, 0.507 in/单位) 映射到 (3.15–9.80, 0.35 in/单位)，
    # 比例 1.415 保证每个方框的物理高度与字号和旧版一致。
    K = 1.415

    def Y(y):
        return 3.15 + (y - 1.60) * K

    # zones
    zone(ax, 0.10, Y(1.60), 3.45, Y(6.30), PALETTE['gray_tint'])
    zone(ax, 3.70, Y(1.60), 11.15, Y(6.30), PALETTE['blue_tint'])
    zone(ax, 11.40, Y(1.60), 14.30, Y(6.30), PALETTE['green_1'])

    badge(ax, 0.42, Y(6.05), 1, 'Inputs', r=0.27)
    badge(ax, 4.02, Y(6.05), 2, 'Zero-training rule pipeline (config F)', r=0.27)
    badge(ax, 11.72, Y(6.05), 3, 'Evaluation', r=0.27)

    # inputs
    box(ax, 1.78, Y(4.75), 3.10, 1.05 * K, 'Sentinel-1 SLC\nsingle pre/post pair', '#FFFFFF')
    box(ax, 1.78, Y(3.00), 3.10, 1.05 * K, '8-band patch\n(512²→256²)', '#FFFFFF')
    arrow(ax, 1.78, Y(4.22), 1.78, Y(3.55))

    # evidence
    box(ax, 5.35, Y(4.75), 2.50, 0.90 * K, 'S_int\n(dark water)', PALETTE['green_2'])
    box(ax, 5.35, Y(3.05), 2.50, 0.90 * K, 'S_chg\n(|post−pre|)', PALETTE['green_2'])
    arrow(ax, 3.35, Y(3.05), 4.08, Y(4.52))
    arrow(ax, 3.35, Y(3.05), 4.08, Y(3.12))

    # gate + mask
    box(ax, 8.45, Y(3.90), 2.90, 1.65 * K,
        'Verification gate\nS_int ∩ S_chg\npost-VV < −12 dB\nconn ≥ 500 px',
        PALETTE['blue_main'], lw=1.6, tc='white')
    arrow(ax, 6.62, Y(4.60), 6.98, Y(4.25))
    arrow(ax, 6.62, Y(3.20), 6.98, Y(3.55))
    box(ax, 10.60, Y(3.90), 1.05, 1.05 * K, 'flood\nmask', PALETTE['green_3'])
    arrow(ax, 9.92, Y(3.90), 10.06, Y(3.90))

    ax.text(4.02, Y(2.32), 'Ablation: A (S_int) → C (routing) → F (gate)',
            fontsize=8, color=BLACK)
    ax.text(4.02, Y(1.92), 'S_coh and the urban branch are not in config F (Section V)',
            fontsize=8, color=GRAY_TXT)

    # evaluation
    box(ax, 12.85, Y(5.05), 2.65, 0.85 * K, 'F1 per GT class\n(open / urban)', '#FFFFFF')
    box(ax, 12.85, Y(3.90), 2.65, 0.85 * K, '3 events (4 scenes,\nJubba merged)', '#FFFFFF')
    box(ax, 12.85, Y(2.75), 2.65, 0.85 * K, 'sensitivity +\nofficial split', '#FFFFFF')
    arrow(ax, 11.15, Y(3.90), 11.50, Y(3.90))

    # ---- supervised control band (red dashed) ----
    zone(ax, 3.70, 2.06, 14.30, 3.00, PALETTE['red_tint'],
         ec=PALETTE['red_strong'], ls='--')
    ax.text(9.00, 2.53,
            'Supervised control (reproduced, same protocol): Unet / Unet++ / '
            'DeepLabV3+ — ResNet-34, trained from scratch',
            ha='center', va='center', fontsize=8, color=BLACK)
    arrow(ax, 13.85, 3.00, 13.35, 4.05, color=PALETTE['red_strong'], ls='--')

    # ---- findings band: the three claims this letter defends ----
    zone(ax, 0.10, 0.12, 14.30, 1.90, PALETTE['gray_tint'])
    chips = [
        (0.15, 4.50, 'Open water: config F wins',
         '0.654 vs 0.555 (Unet++)\n3/3 events, +30/+19/+8%',
         PALETTE['blue_tint'], PALETTE['blue_main']),
        (4.97, 4.50, 'The gate is decisive',
         'A→C→F: 0.218→0.246→0.654\ndeterministic; CNN 0.489–0.555',
         PALETTE['blue_tint'], PALETTE['blue_secondary']),
        (9.79, 4.50, 'Boundary: physical, not tuning',
         'urban F1 0.006–0.619; dark-water\ngate excludes double-bounce',
         PALETTE['red_tint'], PALETTE['red_strong']),
    ]
    for x0, w, title, body, fc, ec in chips:
        zone(ax, x0, 0.26, x0 + w, 1.76, fc, ec=ec)
        ax.text(x0 + w / 2, 1.50, title, ha='center', va='center',
                fontsize=8, color=BLACK, fontweight='bold')
        ax.text(x0 + w / 2, 0.87, body, ha='center', va='center',
                fontsize=7.5, color=BLACK, linespacing=1.35)

    export(fig, 'fig1_architecture_housestyle'); plt.close(fig)

# ================= Fig 2 =================
def fig_gateflow():
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 5.6); ax.axis('off')
    ax.text(7.8, 5.32, 'config F: pixel decision flow', ha='center', fontsize=9, color=BLACK)

    xs = [1.15, 3.35, 5.55, 7.75, 9.95, 12.15, 14.35]
    y = 4.00
    box(ax, xs[0], y, 2.00, 1.00, '8-band\npatch', PALETTE['neutral'])
    box(ax, xs[1], y, 2.00, 1.00, 'per-scene\nOtsu', PALETTE['green_1'])
    box(ax, xs[2], y + 0.55, 2.00, 0.80, 'S_int', PALETTE['green_2'])
    box(ax, xs[2], y - 0.55, 2.00, 0.80, 'S_chg', PALETTE['green_2'])
    box(ax, xs[3], y, 2.00, 1.00, 'intersection\n(2 signals)', PALETTE['blue_main'],
        lw=1.5, tc='white')
    box(ax, xs[4], y, 2.00, 1.00, 'post-VV\n< −12 dB', PALETTE['blue_secondary'])
    box(ax, xs[5], y, 2.00, 1.00, 'conn\n≥ 500 px', PALETTE['blue_secondary'])
    box(ax, xs[6], y, 1.80, 1.00, 'flood\nmask', PALETTE['green_3'])

    arrow(ax, 2.20, y, 2.30, y)
    arrow(ax, 4.40, y, 4.50, y + 0.52); arrow(ax, 4.40, y, 4.50, y - 0.52)
    arrow(ax, 6.60, y + 0.55, 6.70, y + 0.27); arrow(ax, 6.60, y - 0.55, 6.70, y - 0.27)
    arrow(ax, 8.80, y, 8.90, y); arrow(ax, 11.00, y, 11.10, y); arrow(ax, 13.20, y, 13.40, y)

    for cx, sx, text in ((6.85, 7.75, 'permanent water\nnon-flood change'),
                         (9.95, 9.95, 'mid-dark\nnon-water'),
                         (13.05, 12.15, 'speckle')):
        arrow(ax, sx, 3.42, cx, 2.95, color=GRAY_TXT, ls='--')
        box(ax, cx, 2.10, 2.85, 1.45, text, PALETTE['red_1'],
            ec=PALETTE['red_strong'], ls='--', fs=7.5)

    ax.text(7.8, 0.72, 'dashed red = false-positive classes removed by that constraint',
            ha='center', fontsize=8, color=GRAY_TXT)
    ax.text(7.8, 0.25, 'S_coh and the urban branch are not part of config F.',
            ha='center', fontsize=8, color=BLACK)
    export(fig, 'fig2_gate_flow_housestyle'); plt.close(fig)


# ================= Fig 2 (single-column vertical) =================
def fig_gateflow_col_v1():
    fig, ax = plt.subplots(figsize=(3.5, 4.3))
    ax.set_xlim(0, 7.0); ax.set_ylim(0, 11.4); ax.axis('off')
    ax.text(3.5, 11.1, 'config F: pixel decision flow', ha='center', fontsize=9, color=BLACK)

    W = 6.0
    box(ax, 3.5, 10.30, W, 0.80, '8-band patch (256²)', PALETTE['neutral'])
    arrow(ax, 3.5, 9.90, 3.5, 9.55)
    box(ax, 3.5, 9.10, W, 0.80, 'per-scene Otsu thresholds', PALETTE['green_1'])
    arrow(ax, 3.5, 8.70, 2.05, 8.35)
    arrow(ax, 3.5, 8.70, 4.95, 8.35)
    box(ax, 2.05, 7.90, 3.10, 0.85, 'S_int\n(dark water)', PALETTE['green_2'], fs=7.5)
    box(ax, 4.95, 7.90, 3.10, 0.85, 'S_chg\n(|post−pre|)', PALETTE['green_2'], fs=7.5)
    arrow(ax, 2.05, 7.48, 2.70, 7.05)
    arrow(ax, 4.95, 7.48, 4.30, 7.05)
    box(ax, 3.5, 6.35, W, 1.30,
        'intersection of two signals\nremoves permanent water\nand non-flood change',
        PALETTE['blue_main'], lw=1.5, tc='white', fs=7.5)
    arrow(ax, 3.5, 5.70, 3.5, 5.40)
    box(ax, 3.5, 4.85, W, 1.05,
        'post-VV < −12 dB\nremoves mid-dark non-water (wet soil)',
        PALETTE['blue_secondary'], fs=7.5)
    arrow(ax, 3.5, 4.33, 3.5, 4.03)
    box(ax, 3.5, 3.45, W, 1.05,
        'connected components ≥ 500 px\nremoves speckle',
        PALETTE['blue_secondary'], fs=7.5)
    arrow(ax, 3.5, 2.93, 3.5, 2.63)
    box(ax, 3.5, 2.20, W, 0.80, 'flood mask', PALETTE['green_3'])
    ax.text(3.5, 1.35, 'S_coh and the urban branch are not\npart of config F (Section V).',
            ha='center', va='center', fontsize=7.5, color=GRAY_TXT)
    export(fig, 'fig2_gate_flow_col_v1', pad=0.0); plt.close(fig)


# ================= Fig 3 =================
model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None,
                         in_channels=8, classes=3).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()


def get_patch(sc, r, c):
    with rasterio.open(os.path.join(PATCHES, sc, f'{sc}_ID_{r}_{c}_SAR.tif')) as ds:
        sar = ds.read().astype(np.float32)
    with rasterio.open(os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')) as ds:
        gt = ds.read(1)[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
    return sar, gt


def config_f(sar):
    s = all_strategies(sar)
    return conn_filter(s['int'] & s['chg'] & (np.nan_to_num(sar[7]) < POST_THR), CONN_MIN)


def cnn_flood(sar):
    with torch.no_grad():
        p = model(torch.from_numpy(normalize(sar)).unsqueeze(0).to(device))
    return p.argmax(1).squeeze(0).cpu().numpy() > 0


def gray_bg(sar):
    a = 0.6 * np.nan_to_num(sar[7], nan=-20.0) + 0.4 * np.nan_to_num(sar[6], nan=-20.0)
    lo, hi = np.percentile(a, [2, 98])
    return np.stack([np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)] * 3, axis=-1)


def tint(bg, mask, color, alpha=0.6):
    out = bg.copy()
    rgb = np.array(matplotlib.colors.to_rgb(color))
    for i in range(3):
        ch = out[..., i]
        ch[mask] = ch[mask] * (1 - alpha) + rgb[i] * alpha
    return out


def pick_jubba2(sc='20231201_Jubba_2', min_px=100):
    with rasterio.open(os.path.join(DATA, sc, 'testing_case_orig', sc, f'{sc}_GT.tif')) as ds:
        gtf = ds.read(1)
    best = None
    for f in glob.glob(os.path.join(PATCHES, sc, '*_SAR.tif')):
        m = re.search(r'ID_(\d+)_(\d+)_SAR', os.path.basename(f))
        r, c = int(m.group(1)), int(m.group(2))
        n = int((gtf[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] > 0).sum())
        if n >= min_px and (best is None or n > best[2]):
            best = (r, c, n)
    return best


def fig_masks():
    b = pick_jubba2()
    scenes = [('20210727_Weihui', 16, 35, 'Weihui (16,35)'),
              ('20231201_Jubba_1', 22, 46, 'Jubba_1 (22,46)')]
    if b:
        scenes.append(('20231201_Jubba_2', b[0], b[1], f'Jubba_2 ({b[0]},{b[1]})'))

    fig, axes = plt.subplots(len(scenes), 4, figsize=(7.16, 1.85 * len(scenes)))
    names = ['SAR (post VV/VH)', 'Ground truth', 'config F (rule)', 'CNN Unet++']
    for row, (sc, r, c, title) in enumerate(scenes):
        sar, gt = get_patch(sc, r, c)
        bg = gray_bg(sar)
        cf, cnn = config_f(sar), cnn_flood(sar)
        print(f'{title}: GT={int((gt > 0).sum())} F={int(cf.sum())} CNN={int(cnn.sum())}')

        gtp = tint(bg * 0.8, gt > 0, PALETTE['highlight'], alpha=0.5)
        gtp = tint(gtp, gt == 2, PALETTE['red_strong'], alpha=0.9)
        panels = [bg, gtp,
                  tint(bg * 0.8, cf, PALETTE['blue_main'], alpha=0.6),
                  tint(bg * 0.8, cnn, PALETTE['red_2'], alpha=0.6)]
        for col, (im, nm) in enumerate(zip(panels, names)):
            a = axes[row, col]
            a.imshow(np.clip(im, 0, 1))
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(0.8); s.set_color(BLACK)
            if row == 0:
                a.set_title(nm, fontsize=8)
            if col == 0:
                a.set_ylabel(title, fontsize=8)

    fig.legend(handles=[
        Patch(facecolor=PALETTE['highlight'], edgecolor=BLACK, label='GT flooded area'),
        Patch(facecolor=PALETTE['red_strong'], edgecolor=BLACK, label='GT urban flood'),
        Patch(facecolor=PALETTE['blue_main'], edgecolor=BLACK, label='config F (rule mask)'),
        Patch(facecolor=PALETTE['red_2'], edgecolor=BLACK, label='CNN Unet++ (pred > 0)'),
    ], loc='lower center', ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1], pad=1)
    export(fig, 'fig3_masks_3row'); plt.close(fig)


# ================= Fig 1b: single-column overview (page-1 lead figure) =================
def fig_overview_bubbles():
    """首页总览图：行对比示意（开阔水 ✓ / 城市 ✗），真实 SAR 缩略图 + 判据气泡 + 结论脚注。"""
    import numpy as _np

    W, H = 252, 268
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    texts, boxes = [], []
    P, PW = 74.0, 66.0
    XA, XB, XC = 44.0, 186.0, 115.0

    def band(y0, y1, fc):
        boxes.append(ax.add_patch(FancyBboxPatch((4, y0), W - 8, y1 - y0,
                     boxstyle='round,pad=0,rounding_size=4', linewidth=0.8,
                     edgecolor='#B8B8B8', facecolor=fc, zorder=0)))

    def bubble(y, text, ec):
        boxes.append(ax.add_patch(FancyBboxPatch((XC - PW / 2, y - 16), PW, 32,
                     boxstyle='round,pad=0,rounding_size=3', linewidth=0.9,
                     edgecolor=ec, facecolor='#FFFFFF', zorder=3)))
        texts.append(ax.text(XC, y, text, ha='center', va='center', fontsize=6.5,
                             color=BLACK, zorder=4, linespacing=1.45))

    def panel(x, y, img, ec, title):
        iax = ax.inset_axes([x - P / 2, y - P / 2, P, P], transform=ax.transData)
        iax.imshow(_np.clip(img, 0, 1))
        iax.set_xticks([]); iax.set_yticks([])
        for s in iax.spines.values():
            s.set_linewidth(1.0); s.set_color(ec)
        texts.append(ax.text(x, y - P / 2 - 7, title, ha='center', va='center',
                             fontsize=6.5, color=BLACK, zorder=4))
        return iax

    def mark(iax, ok):
        """把勾/叉标记画在预测面板的右上角（图像坐标系）。"""
        iax.add_patch(Circle((214, 42), 23, facecolor=(PALETTE['green_3'] if ok else PALETTE['red_2']),
                             edgecolor=('#3F6B3F' if ok else PALETTE['red_strong']),
                             linewidth=1.0, zorder=6))
        iax.text(214, 42, '\u2713' if ok else '\u2717', ha='center', va='center',
                 fontsize=10, color=(BLACK if ok else 'white'), zorder=7)

    for (sc, r, c, ok), (b0, b1, yc, hy) in zip(
            [('20231201_Jubba_1', 23, 48, True), ('20210727_Weihui', 13, 31, False)],
            [(164.0, 252.0, 212.0, 258.0), (58.0, 146.0, 106.0, 152.0)]):
        band(b0, b1, PALETTE['blue_tint'] if ok else PALETTE['red_tint'])
        texts.append(ax.text(8, hy, 'Open water \u2014 the gate commits' if ok
                             else 'Urban \u2014 the physical premise fails',
                             ha='left', va='center', fontsize=8, color=BLACK, fontweight='bold'))

        sar, gt = get_patch(sc, r, c)
        bg = gray_bg(sar)
        panel(XA, yc, bg, BLACK, 'post-VV')

        if ok:
            shown, gt_mask, gc = tint(bg * 0.85, config_f(sar), PALETTE['blue_main'], alpha=0.6), \
                (gt == 1).astype(float), PALETTE['highlight']
            bubble(yc, 'water is dark\nand changed\n\u2192 gate passes', PALETTE['blue_main'])
            title, ec = 'config F mask', PALETTE['green_3']
        else:
            shown, gt_mask, gc = bg * 0.85, (gt == 2).astype(float), PALETTE['red_strong']
            bubble(yc, 'buildings are bright\n(double-bounce)\n\u2192 gate fails',
                   PALETTE['red_strong'])
            title, ec = 'config F mask (empty)', PALETTE['red_strong']

        iax = panel(XB, yc, shown, ec, title)
        if gt_mask.max() > 0:
            iax.contour(gt_mask, levels=[0.5], colors=[gc], linewidths=0.9)

        mark(iax, ok)

    # 图例：色块 + 短标签
    for x, fc, ec, label in ((64, PALETTE['blue_main'], PALETTE['blue_main'], 'config F mask'),
                             (126, 'none', PALETTE['highlight'], 'GT open water'),
                             (188, 'none', PALETTE['red_strong'], 'GT urban flood')):
        ax.add_patch(FancyBboxPatch((x - 30, 40.5), 7, 7, boxstyle='round,pad=0,rounding_size=1',
                                    linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=3))
        texts.append(ax.text(x - 20, 44, label, ha='left', va='center',
                             fontsize=6.5, color=BLACK, zorder=4))

    for y, s, fs, w in ((30, 'config F 0.654 vs CNN 0.555 \u2014 3/3 events', 7.0, 'bold'),
                        (19, 'gain is in the gate: A\u2192C\u2192F 0.218\u21920.246\u21920.654', 7.0, 'bold'),
                        (8, 'boundary: urban F1 0.006\u20130.619 \u2014 physical, not tuning', 7.0, 'normal')):
        texts.append(ax.text(126, y, s, ha='center', va='center', fontsize=fs,
                             color=BLACK, fontweight=w, zorder=4))

    # ---------- 自检：文字不越框、元素不越画布 ----------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi
    bad = 0
    canvas = ax.get_window_extent(rend)
    for tx in texts:
        tb = tx.get_window_extent(rend)
        if tb.x0 < canvas.x0 - 1 or tb.x1 > canvas.x1 + 1 or tb.y0 < canvas.y0 - 1 or tb.y1 > canvas.y1 + 1:
            bad += 1
            print('  [OUT OF CANVAS]', repr(tx.get_text()[:28]))
        cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
        cand = [b.get_window_extent(rend) for b in boxes
                if b.get_window_extent(rend).x0 <= cx <= b.get_window_extent(rend).x1
                and b.get_window_extent(rend).y0 <= cy <= b.get_window_extent(rend).y1]
        if cand:
            bb = min(cand, key=lambda r_: r_.width * r_.height)
            m = (tb.x0 - bb.x0, bb.x1 - tb.x1, tb.y0 - bb.y0, bb.y1 - tb.y1)
            if min(m) < 1:
                bad += 1
                print(f'  [OVERFLOW BOX] {tx.get_text()[:24]!r}: '
                      f'L{m[0]*k:.1f} R{m[1]*k:.1f} T{m[2]*k:.1f} B{m[3]*k:.1f} pt')
    print(f'  layout check: {bad} issue(s) / {len(texts)} texts')

    export(fig, 'fig1_overview_v1_bubbles', pad=0.0); plt.close(fig)


# ================= Fig 1b: single-column overview (page-1 lead figure) =================
def fig_overview_col():
    """首页总览图（B 版）：行对比 + 矩形说明框 + 角标勾/叉 + 每行一句操作结论。"""
    import numpy as _np

    W, H = 252, 292
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    texts, boxes = [], []
    P, BW = 74.0, 66.0
    XA, XB, XC = 44.0, 208.0, 126.0   # 面板/说明框/面板，间距 12 pt，左右边距 7 pt

    def band(y0, y1, fc):
        boxes.append(ax.add_patch(FancyBboxPatch((4, y0), W - 8, y1 - y0,
                     boxstyle='round,pad=0,rounding_size=4', linewidth=0.8,
                     edgecolor='#B8B8B8', facecolor=fc, zorder=0)))

    def note(y, text, ec):
        boxes.append(ax.add_patch(FancyBboxPatch((XC - BW / 2, y - 16), BW, 32,
                     boxstyle='round,pad=0,rounding_size=2', linewidth=0.8,
                     edgecolor=ec, facecolor='#FFFFFF', zorder=3)))
        texts.append(ax.text(XC, y, text, ha='center', va='center', fontsize=7,
                             color=BLACK, zorder=4, linespacing=1.45))

    def panel(x, y, img, ec, title):
        iax = ax.inset_axes([x - P / 2, y - P / 2, P, P], transform=ax.transData)
        iax.imshow(_np.clip(img, 0, 1))
        iax.set_xticks([]); iax.set_yticks([])
        for s in iax.spines.values():
            s.set_linewidth(1.0); s.set_color(ec)
        texts.append(ax.text(x, y - P / 2 - 5, title, ha='center', va='center',
                             fontsize=7, color=BLACK, zorder=4))
        return iax

    def mark(iax, ok):
        """角标：小圆 + 勾/叉（图像坐标系右上角）。"""
        c = '#3F6B3F' if ok else PALETTE['red_strong']
        iax.add_patch(Circle((222, 34), 13, facecolor='#FFFFFF', edgecolor=c,
                             linewidth=0.9, zorder=6))
        iax.text(222, 34, '\u2713' if ok else '\u2717', ha='center', va='center',
                 fontsize=8, color=c, zorder=7)

    rows = [('20231201_Jubba_1', 23, 48, True, 240.0, 284.0, 182.0, 278.0, 196.0, 187.0,
             'dark water and\nintensity change\n\u2192 gate holds',
             'No labels, no training: the rule commits the open-water flood.'),
            ('20210727_Weihui', 13, 31, False, 128.0, 172.0, 70.0, 166.0, 84.0, 75.0,
             'buildings are\nbright, so the\ndark bound fails',
             'Same rule, same thresholds: nothing committed \u2014 context-defined target.')]

    for sc, r, c, ok, yc, hy, b0, b1, cy, ty, notetxt, take in rows:
        band(b0, b1, PALETTE['blue_tint'] if ok else PALETTE['red_tint'])
        texts.append(ax.text(8, hy, 'Open water \u2014 the gate holds' if ok
                             else 'Urban \u2014 the physical premise fails',
                             ha='left', va='center', fontsize=8, color=BLACK, fontweight='normal'))
        sar, gt = get_patch(sc, r, c)
        bg = gray_bg(sar)
        panel(XA, yc, bg, BLACK, 'post-VV')

        if ok:
            shown, gt_mask, gc = tint(bg * 0.85, config_f(sar), PALETTE['blue_main'], alpha=0.6), \
                (gt == 1).astype(float), PALETTE['highlight']
            ec, title = PALETTE['green_3'], 'config F mask'
        else:
            shown, gt_mask, gc = bg * 0.85, (gt == 2).astype(float), PALETTE['red_strong']
            ec, title = PALETTE['red_strong'], 'config F mask (empty)'

        note(yc, notetxt, PALETTE['blue_main'] if ok else PALETTE['red_strong'])
        iax = panel(XB, yc, shown, ec, title)
        if gt_mask.max() > 0:
            iax.contour(gt_mask, levels=[0.5], colors=[gc], linewidths=0.9)
        mark(iax, ok)
        texts.append(ax.text(126, ty, take, ha='center', va='center',
                             fontsize=7, color=GRAY_TXT, zorder=4))

    # 图例（四项等距，左右边界与上方面板外缘对齐：x = 7 … 245）
    for x, fc, ec, ls, label in ((7, '#C8622C', '#C8622C', '-', 'adaptive'),
                                 (66, PALETTE['blue_main'], PALETTE['blue_main'], '-', 'fixed \u2192 mask'),
                                 (125, 'none', PALETTE['highlight'], '-', 'GT water'),
                                 (184, 'none', PALETTE['red_strong'], '-', 'GT urban')):
        ax.add_patch(FancyBboxPatch((x, 52.5), 7, 7, boxstyle='round,pad=0,rounding_size=1',
                                    linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=3))
        texts.append(ax.text(x + 9, 56, label, ha='left', va='center',
                             fontsize=6.5, color=BLACK, zorder=4))

    # ---- F1 阶梯条（数量条 + 花括号，借 main-4 Fig.3 手法）----
    B0, B1, BY, BH = 7.0, 245.0, 38.0, 10.0
    V0, V1 = 0.218, 0.654

    def xv(v):
        return B0 + (v - V0) / (V1 - V0) * (B1 - B0)

    steps = [(0.218, 0.288, '#C8622C', '0.288'),
             (0.288, 0.435, PALETTE['blue_main'], '0.435'),
             (0.435, 0.654, PALETTE['blue_main'], '0.654')]
    for k, (a, b, col, lab) in enumerate(steps):
        x0, x1 = xv(a) + (1.9 if k else 0.0), xv(b) - (1.9 if k < len(steps) - 1 else 0.0)
        ax.add_patch(FancyBboxPatch((x0, BY), x1 - x0, BH,
                     boxstyle='round,pad=0,rounding_size=1', linewidth=0.8,
                     edgecolor=BLACK, facecolor=col, zorder=3))
        texts.append(ax.text((xv(a) + xv(b)) / 2, BY + BH / 2, lab, ha='center', va='center',
                             fontsize=6.2, color='white', zorder=4))
    for v, lab, dx in ((0.288, '+$S_{\\mathrm{chg}}$', 5.0), (0.435, '+$\\theta$', 0.0), (0.654, '+CC', -4.0)):
        texts.append(ax.text(xv(v) + dx, BY - 7, lab, ha='center', va='center',
                             fontsize=6.2, color=GRAY_TXT, zorder=4))
    texts.append(ax.text(B0, BY - 7, 'A 0.218', ha='left', va='center',
                         fontsize=6.2, color=GRAY_TXT, zorder=4))
    # 花括号：标出闸门带来的增益段
    bx0, bx1, by = xv(0.288), xv(0.654), BY - 15
    ax.plot([bx0, bx0, bx1, bx1], [by + 4, by, by, by + 4], color=GRAY_TXT,
            linewidth=0.8, solid_capstyle='butt', zorder=3)
    texts.append(ax.text((bx0 + bx1) / 2, by - 5, 'the gain is in the gate',
                         ha='center', va='center', fontsize=6.5, color=GRAY_TXT, zorder=4))
    texts.append(ax.text(126, 6, 'config F 0.654 vs CNN 0.555 \u2014 3/3 events',
                         ha='center', va='center', fontsize=7, color=BLACK,
                         fontweight='bold', zorder=4))

    # ---------- 自检 ----------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi
    bad = 0
    canvas = ax.get_window_extent(rend)
    for tx in texts:
        tb = tx.get_window_extent(rend)
        if (tb.x0 < canvas.x0 - 1 or tb.x1 > canvas.x1 + 1
                or tb.y0 < canvas.y0 - 1 or tb.y1 > canvas.y1 + 1):
            bad += 1
            print('  [OUT OF CANVAS]', repr(tx.get_text()[:28]))
        cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
        cand = [b.get_window_extent(rend) for b in boxes
                if b.get_window_extent(rend).x0 <= cx <= b.get_window_extent(rend).x1
                and b.get_window_extent(rend).y0 <= cy <= b.get_window_extent(rend).y1]
        if cand:
            bb = min(cand, key=lambda r_: r_.width * r_.height)
            m = (tb.x0 - bb.x0, bb.x1 - tb.x1, tb.y0 - bb.y0, bb.y1 - tb.y1)
            if min(m) < 1:
                bad += 1
                print(f'  [OVERFLOW BOX] {tx.get_text()[:24]!r}: '
                      f'L{m[0]*k:.1f} R{m[1]*k:.1f} T{m[2]*k:.1f} B{m[3]*k:.1f} pt')
    print(f'  layout check: {bad} issue(s) / {len(texts)} texts')

    export(fig, 'fig1_overview', pad=0.0); plt.close(fig)


# ================= Fig 2 (v2): two-stage decision flow =================
# ================= Fig 2 (wide): icon/glyph architecture diagram =================
def fig_gateflow_wide():
    """Fig.2 v8：三段式横向主流程 + 底部消融条。

    版式约定（1 单位 = 1 pt，fig.add_axes([0,0,1,1])）：
      - 三个等高等宽的圆角面板并排：自适应证据 / 固定闸门 / 已提交掩膜；
      - 左侧两张真实缩略图作为输入，箭头从两者之间进入；
      - 每个面板内模块上下两行等宽对齐，底部一行只保留一条符号链（不重复画模块）；
      - 消融阶梯压成最底一条细带，数值与步骤名写在同一格；
      - 图例并入图注，顶部不再占一整条。
    """
    import numpy as _np
    from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch, Polygon

    W, H = 516, 146
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    texts, boxes = [], []

    def rbox(x0, y0, x1, y1, fc='#FFFFFF', ec='#9A9A9A', lw=0.8, ls='-', rs=3, z=1):
        boxes.append(ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                     boxstyle='round,pad=0,rounding_size=' + str(rs), linewidth=lw,
                     edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z)))

    def txt(x, y, s, fs=6.2, ha='center', va='center', color=BLACK, weight='normal', z=6):
        t = ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, fontweight=weight,
                    zorder=z, linespacing=1.22)
        texts.append(t)
        return t

    def straight(x1, y1, x2, y2, ms=5.0, lw=0.9, color=BLACK, ls='-'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms,
                     linewidth=lw, color=color, linestyle=ls, zorder=2))

    def grid_glyph(xc, yc, size, mask, fc, n=6, ec='#8C8C8C'):
        """真实掩膜降采样成 n×n 像素格。"""
        h, w = mask.shape
        small = _np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                blk = mask[int(i * h / n):int((i + 1) * h / n), int(j * w / n):int((j + 1) * w / n)]
                small[i, j] = 1.0 if (blk.size and blk.mean() > 0.35) else 0.0
        iax = ax.inset_axes([xc - size / 2, yc - size / 2, size, size], transform=ax.transData)
        iax.imshow(small, cmap=matplotlib.colors.ListedColormap(['#FFFFFF', fc]),
                   vmin=0, vmax=1, interpolation='nearest')
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_linewidth(0.7); sp.set_color(ec)

    def icon_hist(xc, yc, col, s=1.0, thr=None):
        """直方图图标（单块阶梯多边形）。"""
        hs = (4.0, 7.0, 5.0, 8.0)
        w = 2.6 * s
        pts = [(xc - 5 * s, yc - 4 * s)]
        for i, hh in enumerate(hs):
            x0 = xc - 5 * s + i * w
            pts += [(x0, yc - 4 * s), (x0, yc - 4 * s + hh * s),
                    (x0 + 1.7 * s, yc - 4 * s + hh * s), (x0 + 1.7 * s, yc - 4 * s)]
        pts.append((xc + 5 * s + w * 0.2, yc - 4 * s))
        ax.add_patch(Polygon(pts, closed=True, facecolor=col, edgecolor='none', zorder=4))
        ax.plot([xc - 5.6 * s, xc + 5.6 * s], [yc + 4.4 * s, yc + 4.4 * s], color=col,
                lw=0.7, ls=(0, (2, 2)), zorder=4)
        if thr is not None:
            xth = xc - 5 * s + thr * (7 * s + 5 * s)
            ax.plot([xth, xth], [yc - 4.2 * s, yc + 3.4 * s], color=col, lw=0.7,
                    ls=(0, (1.2, 1.2)), zorder=5)

    def icon_level(xc, yc, col, s=1.0):
        ax.add_patch(Rectangle((xc - 6 * s, yc - 4 * s), 12 * s, 8 * s, facecolor=col,
                               edgecolor='none', alpha=0.25, zorder=4))
        ax.plot([xc - 6 * s, xc + 6 * s], [yc - 1 * s, yc - 1 * s], color=col, lw=0.9, zorder=4)

    def icon_blobs(xc, yc, col, s=1.0):
        pts = [(-5.5, -1.0), (-4.0, 2.2), (-1.0, 3.0), (1.2, 1.4), (3.4, 2.3),
               (6.0, 0.3), (4.8, -2.2), (1.7, -3.0), (-1.3, -2.6), (-3.6, -2.8)]
        ax.add_patch(Polygon([(xc + px * s, yc + py * s) for px, py in pts], closed=True,
                             facecolor=col, edgecolor='none', zorder=4))

    def module(x0, x1, yc, title, icon, col, fc='#FFFFFF', h=20, sub=''):
        rbox(x0, yc - h / 2, x1, yc + h / 2, fc, ec=col, lw=0.9)
        icon(x0 + 12, yc, col, 0.95)
        txt(x0 + 23, yc + (2.6 if sub else 0.0), title, fs=6.8, ha='left')
        if sub:
            txt(x0 + 23, yc - 6.0, sub, fs=5.6, ha='left', color=GRAY_TXT)

    # ---------------- 数据 ----------------
    sar, gt = get_patch('20210727_Weihui', 24, 32)
    s = all_strategies(sar)
    cand = s['int'] & s['chg']
    fin = config_f(sar)
    ORANGE, BLUE, GRAY = '#C8622C', PALETTE['blue_main'], '#8C8C8C'

    post_img = gray_bg(sar)
    dI = _np.abs(_np.nan_to_num(sar[7], nan=-20.0) - _np.nan_to_num(sar[4], nan=-20.0))
    lo_d, hi_d = _np.percentile(dI, [2, 98])
    delta_img = _np.stack([_np.clip((dI - lo_d) / max(hi_d - lo_d, 1e-6), 0, 1)] * 3, axis=-1)

    # ---------------- 输入：两张真实缩略图 ----------------
    txt(6, 126, 'SLC pair', fs=6.0, ha='left', color=GRAY_TXT, z=3)
    for y0, img, lab in ((96, post_img, 'post'), (66, delta_img, r'$|\Delta I|$')):
        iax = ax.inset_axes([8, y0, 26, 26], transform=ax.transData)
        iax.imshow(_np.clip(img, 0, 1))
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_linewidth(0.6); sp.set_color('#8C8C8C')
        txt(36, y0 + 13, lab, fs=5.2, ha='left', z=3)

    # ---------------- 三个等高面板 ----------------
    PY0, PY1 = 44, 132
    panels = ((52, 248, '#FDF7F1', ORANGE, 'Adaptive evidence, per scene', '#8A4A1E'),
              (254, 402, '#F1F5FB', BLUE, 'Fixed verification, no training', '#123A6B'),
              (408, 510, '#FFFFFF', PALETTE['green_3'], 'Committed mask', PALETTE['green_3']))
    for x0, x1, fc, ec, title, tc in panels:
        rbox(x0, PY0, x1, PY1, fc, ec=ec, lw=1.0 if ec != PALETTE['green_3'] else 1.2, z=0)
        txt(x0 + 5, 126, title, fs=6.4, ha='left', color=tc, weight='bold', z=2)

    straight(48, 94, 52, 94, ms=4.6, lw=0.9)
    straight(248, 97, 254, 97, ms=4.6, lw=0.9)
    straight(402, 97, 408, 97, ms=4.6, lw=0.9)

    # ---- 面板 1：两路证据 + 交集 ----
    module(57, 243, 108, r'$S_{\mathrm{int}}$: post $<\tau_{\mathrm{int}}$', icon_hist, ORANGE, h=19)
    module(57, 243, 84, r'$S_{\mathrm{chg}}$: $|$post$-$pre$| > \tau_{\mathrm{chg}}$', icon_hist, ORANGE, h=19)
    grid_glyph(74, 62, 13, s['int'], '#C9D6E4')
    grid_glyph(104, 62, 13, s['chg'], '#C9D6E4')
    straight(81, 62, 97, 62, ms=4.2, lw=0.8)
    ax.add_patch(Circle((130, 62), 7.0, facecolor='#FFFFFF', edgecolor=BLACK, linewidth=0.8, zorder=4))
    txt(130, 62, r'$\cap$', fs=6.6, z=5)
    straight(112, 62, 122, 62, ms=4.2, lw=0.8)
    straight(138, 62, 190, 62, ms=4.2, lw=0.8)
    grid_glyph(203, 62, 14, cand, '#E9A6A1')
    txt(203, 50, 'candidate', fs=5.4, color=GRAY_TXT, z=3)

    # ---- 面板 2：两个常量 + 判定 ----
    module(259, 397, 108, 'post-VV $< -12$ dB', icon_level, BLUE, fc='#E8F0F8', h=19)
    module(259, 397, 84, r'components $\geq$ 500 px', icon_blobs, BLUE, fc='#E8F0F8', h=19)
    grid_glyph(276, 62, 13, cand, '#E9A6A1')
    straight(284, 62, 320, 62, ms=4.2, lw=0.8)
    grid_glyph(334, 62, 14, fin, BLUE)
    txt(276, 50, 'candidate', fs=5.4, color=GRAY_TXT, z=3)
    txt(334, 50, 'committed', fs=5.4, color=GRAY_TXT, z=3)

    # ---- 面板 3：结果 ----
    txt(459, 104, '0.654', fs=11.5, weight='bold', color=BLUE, z=3)
    txt(459, 88, 'mean open F1', fs=5.8, z=3)
    txt(459, 76, 'vs CNN 0.555', fs=5.8, color=GRAY_TXT, z=3)
    txt(459, 60, 'configuration F', fs=5.8, color=GRAY_TXT, z=3)

    # ---------------- 底部：消融条 ----------------
    # 标签与数值分开绘制、同一基线，四格字号统一，避免 math 下标把行高拉歪
    steps = ((6, 128, r'$\mathrm{A}$', '0.218', '#D8D8D8', BLACK, 0.8),
             (132, 254, r'$+S_{\mathrm{chg}}$', '0.288', '#E8B79A', BLACK, 0.8),
             (258, 380, r'$+\theta$', '0.435', ORANGE, '#FFFFFF', 0.8),
             (384, 510, r'$+\mathrm{CC}$', '0.654', BLUE, '#FFFFFF', 1.0))
    for x0, x1, lab, val, fc, tc, lw in steps:
        rbox(x0, 13, x1, 29, fc, ec=BLACK, lw=lw, z=2)
        cx = (x0 + x1) / 2
        t = txt(cx - 3.0, 21, lab, fs=6.4, ha='right', color=tc, z=3)
        t.set_va('center_baseline')
        t2 = txt(cx + 3.0, 21, val, fs=6.4, ha='left', color=tc, z=3)
        t2.set_va('center_baseline')

    # ---------- 自检 ----------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi
    bad = 0
    canvas = ax.get_window_extent(rend)
    for tx in texts:
        tb = tx.get_window_extent(rend)
        if (tb.x0 < canvas.x0 - 1 or tb.x1 > canvas.x1 + 1
                or tb.y0 < canvas.y0 - 1 or tb.y1 > canvas.y1 + 1):
            bad += 1
            print('  [OUT OF CANVAS]', repr(tx.get_text()[:28]))
        cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
        cands = [b.get_window_extent(rend) for b in boxes
                 if b.get_window_extent(rend).x0 <= cx <= b.get_window_extent(rend).x1
                 and b.get_window_extent(rend).y0 <= cy <= b.get_window_extent(rend).y1]
        if cands:
            bb = min(cands, key=lambda r_: r_.width * r_.height)
            m = (tb.x0 - bb.x0, bb.x1 - tb.x1, tb.y0 - bb.y0, bb.y1 - tb.y1)
            if min(m) < 1:
                bad += 1
                print(f'  [OVERFLOW BOX] {tx.get_text()[:24]!r}: '
                      f'L{m[0]*k:.1f} R{m[1]*k:.1f} T{m[2]*k:.1f} B{m[3]*k:.1f} pt')
    print(f'  layout check: {bad} issue(s) / {len(texts)} texts')

    export(fig, 'fig2_flow_wide', pad=0.0); plt.close(fig)


# ================= Fig 2 (v2): two-stage decision flow =================
# ================= Fig 2 (wide): icon/glyph architecture diagram =================
# ================= Fig 2 (v2): two-stage decision flow =================
def fig_gateflow_col():
    """Fig.2 判定链（阶段版式）：Stage 1 证据策略 → Stage 2 验证闸门。"""
    import numpy as _np

    W, H = 252, 310
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    texts, boxes = [], []

    def stage(y0, y1, tint, title, tag):
        boxes.append(ax.add_patch(FancyBboxPatch((24, y0), W - 28, y1 - y0,
                     boxstyle='round,pad=0,rounding_size=4', linewidth=0.8,
                     edgecolor='#8C8C8C', facecolor=tint, linestyle=(0, (4, 2)), zorder=0)))
        boxes.append(ax.add_patch(FancyBboxPatch((26, y1 - 16), W - 32, 14,
                     boxstyle='round,pad=0,rounding_size=2', linewidth=0.8,
                     edgecolor='#8C8C8C', facecolor='#FFFFFF', zorder=2)))
        texts.append(ax.text(126, y1 - 9, title, ha='center', va='center',
                             fontsize=7.5, color=BLACK, fontweight='bold', zorder=3))
        ax.add_patch(FancyBboxPatch((4, y0), 16, y1 - y0,
                     boxstyle='round,pad=0,rounding_size=3', linewidth=0.8,
                     edgecolor='#8C8C8C', facecolor=tint, zorder=0))
        texts.append(ax.text(12, (y0 + y1) / 2, tag, ha='center', va='center',
                             fontsize=7, color=BLACK, fontweight='bold',
                             rotation=90, zorder=3))

    def node(x, y, w, h, text, fc='#FFFFFF', ec=BLACK, fs=7.0, tc=BLACK):
        boxes.append(ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle='round,pad=0,rounding_size=2', linewidth=0.9,
                     edgecolor=ec, facecolor=fc, zorder=3)))
        texts.append(ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                             color=tc, zorder=4, linespacing=1.35))

    def arr(x1, x2, y):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>',
                     mutation_scale=6.5, linewidth=0.9, color=BLACK, zorder=2))

    def sub(x, y, text):
        texts.append(ax.text(x, y, text, ha='center', va='top', fontsize=6,
                             color=GRAY_TXT, zorder=4, linespacing=1.35))

    # ---- Stage 1：证据策略 ----
    stage(168, 304, PALETTE['blue_tint'],
          'Stage 1 — evidence strategies (adaptive Otsu)', 'Stage 1')
    sar, _gt = get_patch('20210727_Weihui', 24, 32)
    iax = ax.inset_axes([26, 228, 26, 26], transform=ax.transData)
    iax.imshow(_np.clip(gray_bg(sar), 0, 1)); iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_linewidth(0.8); sp.set_color(BLACK)
    arr(54, 61, 241)
    node(90, 241, 54, 34, 'per-scene\nOtsu')
    arr(119, 126, 241)
    node(154, 249, 54, 14, 'S_int', PALETTE['green_2'], fs=6.5)
    node(154, 233, 54, 14, 'S_chg', PALETTE['green_2'], fs=6.5)
    arr(183, 190, 241)
    node(217, 241, 54, 34, 'S_int ∩ S_chg', PALETTE['green_2'])
    sub(43, 220, 'input patch')
    sub(90, 220, 'per-scene Otsu')
    sub(154, 220, 'two independent\nsignals')
    sub(217, 220, 'removes permanent\nwater, non-flood\nchange')

    # ---- 阶段间箭头 ----
    ax.add_patch(FancyArrowPatch((126, 166), (126, 152), arrowstyle='-|>',
                 mutation_scale=8, linewidth=1.2, color=BLACK, zorder=2))

    # ---- Stage 2：验证闸门 ----
    stage(14, 150, PALETTE['green_1'],
          'Stage 2 \u2014 verification gate (fixed constants)', 'Stage 2')
    node(46, 90, 60, 34, 'post-VV\n< \u221212 dB')
    arr(78, 94, 90)
    node(126, 90, 60, 34, 'conn\n\u2265 500 px')
    arr(158, 174, 90)
    node(206, 90, 56, 34, 'flood\nmask', PALETTE['green_3'])
    sub(56, 70, 'removes mid-dark\nnon-water (soil)')
    sub(126, 70, 'removes speckle')
    sub(206, 70, 'committed\nflood')
    texts.append(ax.text(126, 26, 'S_coh and the urban branch are not part of config F',
                         ha='center', va='center', fontsize=6, color=GRAY_TXT, zorder=4))

    # ---------- 自检 ----------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi
    bad = 0
    canvas = ax.get_window_extent(rend)
    for tx in texts:
        tb = tx.get_window_extent(rend)
        if (tb.x0 < canvas.x0 - 1 or tb.x1 > canvas.x1 + 1
                or tb.y0 < canvas.y0 - 1 or tb.y1 > canvas.y1 + 1):
            bad += 1
            print('  [OUT OF CANVAS]', repr(tx.get_text()[:30]))
        cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
        cand = [b.get_window_extent(rend) for b in boxes
                if b.get_window_extent(rend).x0 <= cx <= b.get_window_extent(rend).x1
                and b.get_window_extent(rend).y0 <= cy <= b.get_window_extent(rend).y1]
        if cand:
            bb = min(cand, key=lambda r_: r_.width * r_.height)
            m = (tb.x0 - bb.x0, bb.x1 - tb.x1, tb.y0 - bb.y0, bb.y1 - tb.y1)
            if min(m) < 1:
                bad += 1
                print(f'  [OVERFLOW BOX] {tx.get_text()[:24]!r}: '
                      f'L{m[0]*k:.1f} R{m[1]*k:.1f} T{m[2]*k:.1f} B{m[3]*k:.1f} pt')
    print(f'  layout check: {bad} issue(s) / {len(texts)} texts')

    export(fig, 'fig2_gate_flow_col', pad=0.0); plt.close(fig)


# ================= Fig 2 (wide): icon/glyph architecture diagram =================
# ================= Fig 3 (single column): mask comparison =================
def fig_masks_col():
    """Fig.3 单栏版：3 行（事件）× 3 列（GT / config F / CNN），post-VV SAR 作灰度底图。"""
    import numpy as _np

    W, H = 252, 300
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    texts = []
    P = 72.0
    XS = (58.0, 136.0, 214.0)

    def panel(x, yc, img, title=None, ec=BLACK, lw=0.9):
        iax = ax.inset_axes([x - P / 2, yc - P / 2, P, P], transform=ax.transData)
        iax.imshow(_np.clip(img, 0, 1))
        iax.set_xticks([]); iax.set_yticks([])
        for s in iax.spines.values():
            s.set_linewidth(lw); s.set_color(ec)
        return iax

    for x, t in zip(XS, ('Ground truth', 'config F mask', 'CNN Unet++')):
        texts.append(ax.text(x, 290, t, ha='center', va='center', fontsize=8,
                             color=BLACK, fontweight='bold'))

    scenes = [('20210727_Weihui', 16, 35, 'Weihui (16,35)', 246.0),
              ('20231201_Jubba_1', 23, 48, 'Jubba_1 (23,48)', 162.0),
              ('20231201_Jubba_2', 25, 36, 'Jubba_2 (25,36)', 78.0)]
    for sc, r, c, label, yc in scenes:
        sar, gt = get_patch(sc, r, c)
        bg = gray_bg(sar)
        texts.append(ax.text(11, yc, label, ha='center', va='center', fontsize=6.5,
                             color=BLACK, rotation=90))

        gtp = tint(bg * 0.8, gt > 0, PALETTE['highlight'], alpha=0.5)
        gtp = tint(gtp, gt == 2, PALETTE['red_strong'], alpha=0.9)
        cf = config_f(sar)
        cnn = cnn_flood(sar)
        for x, img, ec in ((XS[0], gtp, BLACK),
                           (XS[1], tint(bg * 0.8, cf, PALETTE['blue_main'], alpha=0.6),
                            PALETTE['green_3']),
                           (XS[2], tint(bg * 0.8, cnn, PALETTE['red_2'], alpha=0.6),
                            PALETTE['red_strong'])):
            panel(x, yc, img, ec=ec)

    # 图例：色块 + 短标签
    items = [(6, PALETTE['highlight'], PALETTE['highlight'], 'GT flooded area'),
             (66, PALETTE['red_strong'], PALETTE['red_strong'], 'GT urban flood'),
             (126, PALETTE['blue_main'], PALETTE['blue_main'], 'config F mask'),
             (186, PALETTE['red_2'], PALETTE['red_strong'], 'CNN Unet++')]
    for x, fc, ec, lab in items:
        ax.add_patch(FancyBboxPatch((x, 15), 7, 7, boxstyle='round,pad=0,rounding_size=1',
                                    linewidth=0.8, edgecolor=ec, facecolor=fc, zorder=3))
        texts.append(ax.text(x + 10, 18.5, lab, ha='left', va='center', fontsize=6.5,
                             color=BLACK, zorder=4))

    # ---------- 自检 ----------
    fig.canvas.draw()
    rend2 = fig.canvas.get_renderer()
    canvas = ax.get_window_extent(rend2)
    bad = 0
    for tx in texts:
        tb = tx.get_window_extent(rend2)
        if (tb.x0 < canvas.x0 - 1 or tb.x1 > canvas.x1 + 1
                or tb.y0 < canvas.y0 - 1 or tb.y1 > canvas.y1 + 1):
            bad += 1
            print('  [OUT OF CANVAS]', repr(tx.get_text()[:28]))
    print(f'  layout check: {bad} issue(s) / {len(texts)} texts')

    export(fig, 'fig3_masks_3col', pad=0.0); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_architecture(); fig_overview_col(); fig_gateflow_col(); fig_masks()
    print('done')
