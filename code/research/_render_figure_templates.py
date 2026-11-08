"""方案 A · 三张图的**线框模板**（wireframe，供手绘/美化底稿）
线框只放**短标签 + 结构**；最终长文案见配套规格说明。
输出（docs/01_论文/）：
  figT1_architecture_wireframe.{png,pdf}
  figT2_gate_flow_wireframe.{png,pdf}
  figT3_mask_grid_wireframe.{png,pdf}
用法：python _render_figure_templates.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

DATA = 'D:/DisasterM3/UrbanSARFloods'
OUT = os.path.join(DATA, 'docs', '01_论文')

WIRE = '#B0B0B0'
FILL = '#F5F5F5'
TXT = '#555555'
NOTE = '#909090'


def wire(ax, x, y, w, h, text, fs=6.5, ls='-'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle='round,pad=0.02,rounding_size=0.06',
                 linewidth=0.9, edgecolor=WIRE, facecolor=FILL,
                 linestyle=ls, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=TXT,
            zorder=3, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, ls='-', color='#AAAAAA'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=9, linewidth=1.0, color=color,
                 linestyle=ls, zorder=1))


def tag(ax, x, y, text, fs=6.0, ha='left'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=NOTE, style='italic')


def save(fig, name):
    fig.savefig(os.path.join(OUT, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, f'{name}.png'), dpi=200, bbox_inches='tight')
    print('saved:', name)


# ------------------------------------------------------------- T1 系统架构
def t1():
    fig, ax = plt.subplots(figsize=(7.16, 4.5))
    ax.set_xlim(-3.2, 13.0); ax.set_ylim(0, 10.0); ax.axis('off')
    tag(ax, -3.1, 9.75, 'TEMPLATE (wireframe) - Figure 1: system architecture')
    ax.text(4.9, 9.35, 'System architecture', ha='center', fontsize=10, color='#333333')

    cols = [0.9, 4.3, 7.7, 11.1]
    rows = [(8.3, 'Data'), (6.7, 'Evidence'), (4.9, 'Decision'), (3.1, 'Evaluation')]
    for y, name in rows:
        ax.text(-3.1, y, name, ha='left', va='center', fontsize=8, color='#333333')

    # Data
    wire(ax, cols[0], 8.3, 3.0, 0.95, 'Sentinel-1 SLC\n(single pair)')
    wire(ax, cols[1], 8.3, 3.0, 0.95, '8 bands\n(coherence+intensity)')
    wire(ax, cols[2], 8.3, 3.0, 0.95, '512²→256²\nofficial split')
    arrow(ax, 2.4, 8.3, 2.8, 8.3); arrow(ax, 5.8, 8.3, 6.2, 8.3)

    # Evidence
    wire(ax, cols[0], 6.7, 3.0, 0.95, 'S_int\n(dark water)')
    wire(ax, cols[1], 6.7, 3.0, 0.95, 'S_chg\n(|post-pre|)')
    wire(ax, cols[2], 6.7, 3.0, 0.95, 'S_coh\n(not in F)', ls='--')
    wire(ax, cols[3], 6.7, 3.0, 0.95, 'urban branch\n(Section V)', ls='--')
    tag(ax, -3.1, 6.0, 'adaptive per-scene Otsu, VV ∪ VH')

    # Decision
    wire(ax, cols[0], 4.9, 3.0, 1.0, 'A\nS_int only')
    wire(ax, cols[1], 4.9, 3.0, 1.0, 'C\nscene routing')
    wire(ax, cols[2], 4.9, 3.0, 1.2, 'F  (reported)\ngate applied')
    wire(ax, cols[3], 4.9, 3.0, 1.0, 'urban branch\n(Section V)', ls='--')

    # Evaluation
    wire(ax, cols[0], 3.1, 3.0, 1.0, 'F1\n(open / urban)')
    wire(ax, cols[1], 3.1, 3.0, 1.0, '3 events\n(4 scenes)')
    wire(ax, cols[2], 3.1, 3.0, 1.0, 'sensitivity +\ncalibration')
    wire(ax, cols[3], 3.1, 3.0, 1.1, 'CNN control\n3 archs, seed 42', ls='--')

    for x in cols:
        arrow(ax, x, 7.8, x, 7.2); arrow(ax, x, 6.2, x, 5.45); arrow(ax, x, 4.35, x, 3.65)

    ax.text(4.9, 1.1, 'Deterministic · no training · constants disclosed',
            ha='center', fontsize=7.5, color='#333333')
    tag(ax, -3.1, 0.35, 'dashed = outside the reported config F')
    save(fig, 'figT1_architecture_wireframe'); plt.close(fig)


# ------------------------------------------------------------- T2 判定流程
def t2():
    fig, ax = plt.subplots(figsize=(7.16, 3.9))
    ax.set_xlim(0, 15.4); ax.set_ylim(0, 6.2); ax.axis('off')
    tag(ax, 0.0, 6.0, 'TEMPLATE (wireframe) - Figure 2: config F decision flow')
    ax.text(7.7, 5.62, 'config F: pixel decision flow', ha='center', fontsize=10, color='#333333')

    xs = [1.2, 3.4, 5.6, 7.8, 10.0, 12.2, 14.4]
    y = 4.3
    wire(ax, xs[0], y, 2.0, 1.0, '8-band\npatch')
    wire(ax, xs[1], y, 2.0, 1.0, 'per-scene\nOtsu')
    wire(ax, xs[2], y + 0.55, 2.0, 0.8, 'S_int')
    wire(ax, xs[2], y - 0.55, 2.0, 0.8, 'S_chg')
    wire(ax, xs[3], y, 2.0, 1.0, 'intersection\n(2 signals)')
    wire(ax, xs[4], y, 2.0, 1.0, 'post-VV\n< -12 dB')
    wire(ax, xs[5], y, 2.0, 1.0, 'connected\n>= 500 px')
    wire(ax, xs[6], y, 1.8, 1.0, 'flood\nmask')

    arrow(ax, 2.25, y, 2.35, y)
    arrow(ax, 4.45, y, 4.55, y + 0.5); arrow(ax, 4.45, y, 4.55, y - 0.5)
    arrow(ax, 6.65, y + 0.55, 6.75, y + 0.28); arrow(ax, 6.65, y - 0.55, 6.75, y - 0.28)
    arrow(ax, 8.85, y, 8.95, y); arrow(ax, 11.05, y, 11.15, y); arrow(ax, 13.25, y, 13.45, y)

    for cx, sx, text in ((6.9, 7.8, 'permanent water\nnon-flood change'),
                         (10.0, 10.0, 'mid-dark\nnon-water'),
                         (13.1, 12.2, 'speckle')):
        arrow(ax, sx, 3.65, cx, 3.15, ls='--')
        wire(ax, cx, 2.35, 2.9, 1.4, text, ls='--', fs=6.2)

    tag(ax, 0.0, 0.55, 'dashed = false positives removed by that constraint')
    ax.text(7.7, 0.15, 'S_coh and the urban branch are not part of config F.',
            ha='center', fontsize=6.3, color='#333333')
    save(fig, 'figT2_gate_flow_wireframe'); plt.close(fig)


# ------------------------------------------------------------- T3 掩膜版面
def t3():
    fig, ax = plt.subplots(figsize=(7.16, 5.6))
    ax.set_xlim(0, 4.4); ax.set_ylim(-0.4, 4.4); ax.axis('off')
    tag(ax, 0.0, 4.25, 'TEMPLATE (wireframe) - Figure 3: mask comparison (paste rasters)')

    headers = ['SAR (post VV/VH)', 'Ground truth', 'config F', 'CNN Unet++']
    for i, h in enumerate(headers):
        ax.text(1.02 + i * 0.95, 3.95, h, ha='center', fontsize=6.5, color='#333333')

    rowlab = ['Weihui', 'Jubba_1', 'Jubba_2 (optional)']
    for j, lab in enumerate(rowlab):
        yy = 3.15 - j * 1.10
        ax.text(0.0, yy, lab, ha='left', va='center', fontsize=5.8, color='#333333')
        for i in range(4):
            wire(ax, 1.02 + i * 0.95, yy, 0.85, 1.0, 'image', fs=5.8, ls='--')

    ax.text(0.0, 0.02, 'Legend:', ha='left', va='center', fontsize=6.0, color='#333333')
    for i, lab in enumerate(['GT open', 'GT urban', 'config F', 'CNN Unet++']):
        ax.add_patch(Rectangle((0.72 + i * 0.9, -0.05), 0.13, 0.13,
                               facecolor='#DDDDDD', edgecolor=WIRE, lw=0.8))
        ax.text(0.88 + i * 0.9, 0.015, lab, fontsize=5.6, color='#5A5A5A', va='center')

    tag(ax, 0.0, -0.30, 'Caption: gray = SAR backscatter, water dark; rule mask is a raw binary map (no smoothing).')
    save(fig, 'figT3_mask_grid_wireframe'); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    t1(); t2(); t3()
    print('templates done ->', OUT)
