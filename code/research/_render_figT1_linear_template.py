"""图1 线性版线框模板（与分层版 figT1_architecture_wireframe 对照比较）
结构：横向线性主干（config F 判定链）+ 下方三条注释带（策略 / 配置 / 评测）
输出：docs/01_论文/figT1L_linear_wireframe.{png,pdf}
用法：python _render_figT1_linear_template.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DATA = 'D:/DisasterM3/UrbanSARFloods'
OUT = os.path.join(DATA, 'docs', '01_论文')

WIRE = '#B0B0B0'
FILL = '#F5F5F5'
BAND = '#EFEFEF'
TXT = '#555555'
NOTE = '#909090'


def box(ax, x, y, w, h, text, fs=6.5, ls='-', fill=FILL, ec=WIRE):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle='round,pad=0.02,rounding_size=0.06',
                 linewidth=0.9, edgecolor=ec, facecolor=fill,
                 linestyle=ls, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=TXT,
            zorder=3, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, ls='-', color='#AAAAAA'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=9, linewidth=1.0, color=color,
                 linestyle=ls, zorder=1))


def band(ax, y, h, label, segs, fs=6.0):
    x0, x1 = 2.3, 15.4
    ax.text(0.05, y, label, ha='left', va='center', fontsize=6.5, color='#333333')
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h,
                 boxstyle='round,pad=0.02,rounding_size=0.06',
                 linewidth=0.9, edgecolor=WIRE, facecolor=BAND, zorder=2))
    n = len(segs)
    w = (x1 - x0) / n
    for k in range(1, n):
        xd = x0 + w * k
        ax.plot([xd, xd], [y - h / 2 + 0.06, y + h / 2 - 0.06],
                color=WIRE, linewidth=0.7, zorder=3)
    for i, s in enumerate(segs):
        ax.text(x0 + w * (i + 0.5), y, s, ha='center', va='center',
                fontsize=fs, color=TXT, zorder=4, linespacing=1.3)


fig, ax = plt.subplots(figsize=(7.16, 3.9))
ax.set_xlim(0, 16.6); ax.set_ylim(0, 6.6); ax.axis('off')
ax.text(0.0, 6.35, 'TEMPLATE (wireframe) - Figure 1 LINEAR variant: backbone + annotation bands',
        ha='left', va='center', fontsize=6.0, color=NOTE, style='italic')
ax.text(8.0, 5.95, 'config F pipeline (linear)', ha='center', fontsize=10, color='#333333')

# ---------------- 主干 ----------------
xs = [1.0, 3.27, 5.53, 7.8, 10.07, 12.33, 14.6]
y = 4.85
labels = ['SLC pair', '8 bands', 'Otsu\nper scene', 'S_int ∩ S_chg',
          'post-VV\n< -12 dB', 'conn\n>= 500 px', 'flood\nmask']
for x, lab in zip(xs, labels):
    box(ax, x, y, 2.0, 0.95, lab)
for i in range(len(xs) - 1):
    arrow(ax, xs[i] + 1.0, y, xs[i + 1] - 1.0, y)

# ---------------- 注释带 ----------------
band(ax, 3.2, 0.85, 'Strategies', 
     ['S_int\ndark water', 'S_chg\n|post-pre|', 'S_coh\n(not in F)', 'urban branch\n(Section V)'])
band(ax, 2.0, 0.85, 'Configurations',
     ['A — S_int only', 'C — scene routing', 'F — intersection + gate\n(reported)'])
band(ax, 0.8, 0.85, 'Evaluation',
     ['F1 per class\n(open / urban)', '3 events · 4 scenes',
      'sensitivity +\ncalibration', 'CNN control\n3 archs, seed 42'])

# 虚线关联（不穿过带体）
arrow(ax, 5.53, 2.65, 7.8, 4.35, ls='--')      # 策略带 -> 交集
arrow(ax, 7.8, 1.6, 10.07, 4.35, ls='--')      # 配置带 -> 门
ax.plot([16.1, 16.1], [0.8, 4.85], color='#AAAAAA', lw=1.0, ls='--', zorder=1)
arrow(ax, 16.1, 4.85, 15.65, 4.85, ls='--')     # 评测带 -> mask（右侧导轨）

ax.text(8.0, 0.15, 'dashed = annotation only (not part of the config-F data flow)',
        ha='center', fontsize=6.0, color=NOTE, style='italic')
fig.savefig(os.path.join(OUT, 'figT1L_linear_wireframe.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'figT1L_linear_wireframe.png'), dpi=200, bbox_inches='tight')
print('saved: figT1L_linear_wireframe')
