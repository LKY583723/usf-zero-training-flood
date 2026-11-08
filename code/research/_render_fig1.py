"""A3 图1：config F 零训练洪水制图管线图（自适应/固定阈值着色）。
输出：UrbanSARFloods/docs/01_论文/fig1_pipeline.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DATA = 'D:/DisasterM3/UrbanSARFloods'


def box(ax, x, y, w, h, text, fc, ec='#333333', fs=10, lw=1.4):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.06',
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, zorder=3, wrap=True)


def arrow(ax, x1, y1, x2, y2, color='#444', lw=1.6):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                        mutation_scale=16, linewidth=lw, color=color, zorder=1)
    ax.add_patch(a)


def dashed_box(ax, x, y, w, h, text, fc='#ffffff', ec='#777777', fs=8):
    """灰色虚线圆角框：标出"存在但被排除"的组件。"""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.06',
                       linewidth=1.3, edgecolor=ec, facecolor=fc,
                       linestyle=(0, (4, 4)), zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            zorder=3, color='#555')


def arrow_dashed(ax, x1, y1, x2, y2, color='#777', lw=1.3):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                        mutation_scale=12, linewidth=lw, color=color,
                        linestyle=(0, (5, 5)), zorder=1)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

GREEN = '#d8f0d8'    # 自适应 (per-patch Otsu)
BLUE = '#d6e6f7'     # 固定阈值 (fixed constants)
YELLOW = '#fdf3cf'   # 输出
GRAY = '#eeeeee'

# 输入
box(ax, 12, 30, 16, 12, '8-band test patch\n256×256\n(raw dB / coherence)', GRAY, fs=9)

# 两个自适应策略
box(ax, 40, 46, 24, 11,
    'S_int — dark water\npost-VV/VH intensity\nper-patch Otsu threshold', GREEN, fs=8.5)
box(ax, 40, 14, 24, 11,
    'S_chg — intensity change\n|post − pre| (VV/VH)\nper-patch Otsu threshold', GREEN, fs=8.5)

# 交集
box(ax, 66, 30, 15, 11,
    'S_int ∩ S_chg\nAND of\nindependent\nevidence', '#fbe9e7', fs=8.5)

# 固定约束
box(ax, 84, 42, 15, 10,
    'post VV\n< −12 dB\n(fixed)', BLUE, fs=8.5)
box(ax, 84, 18, 15, 10,
    'connected\ncomponents\n≥ 500 px\n(fixed)', BLUE, fs=8.5)

# 输出
box(ax, 97, 30, 10, 10, 'flood\nmask', YELLOW, fs=9)

# ---- 新增：相干/城区策略 = 不在 config F（虚线排除组）----
dashed_box(ax, 12, 12, 18, 9,
           'S_coh / S_urban\ncoherence strategies\n(not used in config F)')
arrow_dashed(ax, 12, 23.9, 12, 16.6)
ax.text(12, 6.4, 'dashed = excluded from config F',
        ha='center', fontsize=8, color='#777', style='italic')

arrow(ax, 20.5, 30, 28, 42)
arrow(ax, 20.5, 30, 28, 18)
arrow(ax, 52, 46, 58.6, 33.5)
arrow(ax, 52, 14, 58.6, 26.5)
arrow(ax, 73.5, 30, 76.5, 38)
arrow(ax, 73.5, 30, 76.5, 22)
arrow(ax, 91.5, 42, 91.5, 36)
arrow(ax, 91.5, 18, 91.5, 25)
arrow(ax, 92, 30, 97, 30)

ax.text(40, 54.5, 'adaptive (per-patch Otsu)', ha='center', fontsize=9, color='#2e7d32')
ax.text(84, 50, 'fixed across events', ha='center', fontsize=9, color='#1565c0')
ax.text(66, 39.5, 'verification gate', ha='center', fontsize=9, color='#c62828')
ax.text(50, 6, 'no training · no labels · no seed · fully deterministic',
        ha='center', fontsize=10, style='italic', color='#333')
ax.text(50, 1.5, 'config F  (mean open F1 0.654 on three UrbanSARFloods test events)',
        ha='center', fontsize=9.5, color='#555')

plt.tight_layout()
out = os.path.join(DATA, 'docs', '01_论文', 'fig1_pipeline.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print('saved:', out)
