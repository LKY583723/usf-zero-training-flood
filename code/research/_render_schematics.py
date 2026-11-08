"""矢量图组：图A 物理示意 / 图B 系统架构 / 图C config-F 判定流程
输出（docs/01_论文/）：
  figA_physics_{en,zh}.{pdf,svg,png}
  figB_architecture_{en,zh}.{pdf,svg,png}
  figC_configF_flow_{en,zh}.{pdf,svg,png}
用法：python _render_schematics.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

DATA = 'D:/DisasterM3/UrbanSARFloods'
OUT = os.path.join(DATA, 'docs', '01_论文')

FONTS = {'en': ['DejaVu Sans', 'Arial'],
         'zh': ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']}

TXT = {
# ---------- 图A ----------
'a_title': {'en': 'SAR flood physics and the verification gate',
            'zh': 'SAR 洪水物理与验证门'},
'a_c1': {'en': 'Open\nwater', 'zh': '开阔\n水面'},
'a_c2': {'en': 'Land\n(bare / vegetated)', 'zh': '陆面\n（裸地/植被）'},
'a_c3': {'en': 'Built-up\n(urban)', 'zh': '建成区\n（城区）'},
'a_r1': {'en': 'Pre-event intensity', 'zh': '灾前强度'},
'a_r2': {'en': 'Post-event intensity', 'zh': '灾后强度'},
'a_r3': {'en': 'Δ intensity (post − pre)', 'zh': 'Δ强度（灾后−灾前）'},
'a_r4': {'en': 'Coherence (pre → co)', 'zh': '相干度（灾前→洪水期）'},
'a_p1': {'en': 'mid (dry land)', 'zh': '中等（旱地）'},
'a_p2': {'en': 'mid', 'zh': '中等'},
'a_p3': {'en': 'bright', 'zh': '亮'},
'a_q1': {'en': 'dark', 'zh': '暗'},
'a_q2': {'en': 'mid', 'zh': '中等'},
'a_q3': {'en': 'bright (double\nbounce)', 'zh': '亮（双反弹）'},
'a_d1': {'en': 'large decrease', 'zh': '大幅下降'},
'a_d2': {'en': 'no change', 'zh': '无变化'},
'a_d3': {'en': 'stays bright', 'zh': '保持亮'},
'a_k1': {'en': 'loss (low γ)', 'zh': '失相干（低 γ）'},
'a_k2': {'en': 'stable (high γ)', 'zh': '稳定（高 γ）'},
'a_k3': {'en': 'loss (low γ)', 'zh': '失相干（低 γ）'},
'a_b1': {'en': 'Two independent\nevidence\n→ gate passes',
         'zh': '两条独立证据\n成立\n→ 过门'},
'a_b2': {'en': 'Unchanged\n→ removed by\nS_int ∩ S_chg',
         'zh': '无变化\n→ 被 S_int ∩ S_chg\n清除'},
'a_b3': {'en': 'Bright\n→ excluded by the\ndark-water bound',
         'zh': '变亮\n→ 被暗水约束\n排除'},
'a_foot': {'en': 'Coherence is orthogonal evidence: it is not used by config F '
                '(only by the exploratory urban branch, section V).',
           'zh': '相干度是正交证据：config F 不使用（仅探索性城区分支 §V 使用）。'},
# ---------- 图B ----------
'b_title': {'en': 'Zero-training rule pipeline: system architecture',
            'zh': '零训练规则管线：系统架构'},
'b_layer_data': {'en': 'Data', 'zh': '数据'},
'b_layer_ev': {'en': 'Evidence', 'zh': '证据'},
'b_layer_dec': {'en': 'Decision', 'zh': '决策'},
'b_layer_eval': {'en': 'Evaluation', 'zh': '评测'},
'b_d1': {'en': 'Sentinel-1\nSLC', 'zh': 'Sentinel-1\nSLC'},
'b_d2': {'en': 'Coherence +\nintensity (8 bands)', 'zh': '相干度 +\n强度（8 波段）'},
'b_d3': {'en': 'Samples 512²→256²\nofficial split ·\ntesting_case_256',
         'zh': '样本 512²→256²\n官方划分 ·\ntesting_case_256'},
'b_e1': {'en': 'S_int\ndark water', 'zh': 'S_int\n暗水'},
'b_e2': {'en': 'S_chg\n|post−pre|', 'zh': 'S_chg\n|灾后−灾前|'},
'b_e3': {'en': 'S_coh\ncoherence\n(not in F)', 'zh': 'S_coh\n相干\n（不在 F）'},
'b_e4': {'en': 'S_urban\nexploratory', 'zh': 'S_urban\n探索性'},
'b_e_sub': {'en': 'per-patch Otsu,\nVV ∪ VH', 'zh': '逐场景 Otsu，\nVV ∪ VH'},
'b_c1': {'en': 'A\nS_int only', 'zh': 'A\n仅 S_int'},
'b_c2': {'en': 'C\nscene routing', 'zh': 'C\n场景路由'},
'b_c3': {'en': 'F\nS_int ∩ S_chg\n+ constraints', 'zh': 'F\nS_int ∩ S_chg\n+ 约束'},
'b_c4': {'en': 'F-VLM\nLLM verifier\n(worse)', 'zh': 'F-VLM\nLLM 验证\n（更差）'},
'b_v1': {'en': 'Pixel-aggregated\nF1 (open / urban)', 'zh': '逐类像素聚合\nF1（open / urban）'},
'b_v2': {'en': 'Three-event\nprotocol (4 scenes,\nJubba merged)',
         'zh': '3 事件协议\n（4 场景，\nJubba 合并）'},
'b_v3': {'en': 'Sensitivity sweep +\nofficial-split\ncalibration',
         'zh': '敏感性扫描 +\n官方划分校准'},
'b_v4': {'en': 'CNN reproduction\nUnet / Unet++ / DeepLabV3+\n(SMP ResNet-34, z-score, seed 42)',
         'zh': 'CNN 复现\nUnet / Unet++ / DeepLabV3+\n（SMP ResNet-34, z-score, seed 42）'},
'b_foot': {'en': 'Deterministic · no training · constants disclosed',
           'zh': '确定性 · 零训练 · 常数已披露'},
# ---------- 图C ----------
'c_title': {'en': 'config F: pixel decision flow', 'zh': 'config F：逐像元判定流程'},
'c1': {'en': '8-band\npatch (256²)', 'zh': '8 波段\npatch（256²）'},
'c2': {'en': 'per-scene\nOtsu thresholds', 'zh': '逐场景\nOtsu 阈值'},
'c3': {'en': 'S_int\ndark water', 'zh': 'S_int\n暗水'},
'c4': {'en': 'S_chg\n|post−pre|', 'zh': 'S_chg\n|灾后−灾前|'},
'c5': {'en': 'S_int ∩ S_chg\nindependent\nevidence', 'zh': 'S_int ∩ S_chg\n独立证据\n交集'},
'c6': {'en': 'post-VV\n< −12 dB\n(fixed)', 'zh': 'post-VV\n< −12 dB\n（固定）'},
'c7': {'en': 'connected\ncomponents\n≥ 500 px', 'zh': '连通域\n≥ 500 px\n（固定）'},
'c8': {'en': 'flood\nmask', 'zh': '洪水\n掩膜'},
'c_n1': {'en': 'removes permanent\nwater & non-flood\nchange',
         'zh': '清除永久水体\n与非洪水变化'},
'c_n2': {'en': 'removes mid-dark\nnon-water\n(e.g., wet soil)',
         'zh': '清除中暗非水体\n（如湿土）'},
'c_n3': {'en': 'removes speckle\nfalse positives',
         'zh': '清除斑点\n假阳性'},
'c_foot': {'en': 'S_coh / S_urban are not part of config F (section V).',
           'zh': 'S_coh / S_urban 不参与 config F（§V）。'},
}


def setfont(lang):
    matplotlib.rcParams['font.sans-serif'] = FONTS[lang]
    matplotlib.rcParams['axes.unicode_minus'] = False


def rbox(ax, x, y, w, h, text, fc, ec='#333333', fs=8, lw=1.1, ls='-', tc='#111111'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle='round,pad=0.02,rounding_size=0.06',
                 linewidth=lw, edgecolor=ec, facecolor=fc,
                 linestyle=ls, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc,
            zorder=3, linespacing=1.25)


def arr(ax, x1, y1, x2, y2, color='#444444', lw=1.2, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=12, linewidth=lw, color=color,
                 linestyle=ls, zorder=1))


def save(fig, name, lang):
    for ext in ('pdf', 'svg'):
        fig.savefig(os.path.join(OUT, f'{name}_{lang}.{ext}'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, f'{name}_{lang}.png'), dpi=200, bbox_inches='tight')
    print('saved:', name, lang)


# ------------------------------------------------------------------ 图A
def fig_a(lang):
    setfont(lang)
    T = lambda k: TXT[k][lang]
    fig, ax = plt.subplots(figsize=(7.16, 4.1))
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 9.2); ax.axis('off')
    ax.text(6.6, 8.9, T('a_title'), ha='center', fontsize=10, fontweight='bold')

    cols = [5.4, 8.2, 11.0]
    for x, k in zip(cols, ('a_c1', 'a_c2', 'a_c3')):
        ax.text(x, 8.15, T(k), ha='center', va='center', fontsize=8,
                fontweight='bold', linespacing=1.25)
    for y, k in zip((7.0, 5.85, 4.7, 3.55), ('a_r1', 'a_r2', 'a_r3', 'a_r4')):
        ax.text(0.15, y, T(k), ha='left', va='center', fontsize=8, color='#222222')

    for x, (g, k) in zip(cols, [(0.60, 'a_p1'), (0.60, 'a_p2'), (0.85, 'a_p3')]):
        ax.add_patch(Rectangle((x - 0.8, 6.82), 1.6, 0.40,
                               facecolor=str(g), edgecolor='#888888', lw=0.8))
        ax.text(x, 6.50, T(k), ha='center', fontsize=7.2, color='#333333')

    for x, (g, k) in zip(cols, [(0.18, 'a_q1'), (0.60, 'a_q2'), (0.85, 'a_q3')]):
        ax.add_patch(Rectangle((x - 0.8, 5.67), 1.6, 0.40,
                               facecolor=str(g), edgecolor='#888888', lw=0.8))
        ax.text(x, 5.33, T(k), ha='center', fontsize=7.2, color='#333333')

    ax.annotate('', xy=(cols[0], 4.46), xytext=(cols[0], 5.06),
                arrowprops=dict(arrowstyle='-|>', color='#2e7d32', lw=1.6))
    ax.plot([cols[1] - 0.5, cols[1] + 0.5], [4.76, 4.76], color='#777777', lw=1.6)
    ax.plot([cols[2] - 0.5, cols[2] + 0.5], [4.76, 4.76], color='#e65100', lw=1.6)
    for x, k in zip(cols, ('a_d1', 'a_d2', 'a_d3')):
        ax.text(x, 4.38, T(k), ha='center', fontsize=7.2, color='#333333')

    for x, (g_pre, g_co, k) in zip(cols, [(0.85, 0.25, 'a_k1'),
                                          (0.85, 0.85, 'a_k2'),
                                          (0.85, 0.25, 'a_k3')]):
        ax.add_patch(Rectangle((x - 0.62, 3.45), 0.48, 0.40,
                               facecolor=str(g_pre), edgecolor='#888888', lw=0.8))
        ax.add_patch(Rectangle((x + 0.14, 3.45), 0.48, 0.40,
                               facecolor=str(g_co), edgecolor='#888888', lw=0.8))
        ax.text(x - 0.38, 3.93, 'pre', ha='center', fontsize=5.5, color='#666666')
        ax.text(x + 0.38, 3.93, 'co', ha='center', fontsize=5.5, color='#666666')
        ax.text(x, 3.20, T(k), ha='center', fontsize=7.2, color='#333333')

    for x, (fc, k) in zip(cols, [('#e6f4e6', 'a_b1'), ('#eeeeee', 'a_b2'), ('#fde7e7', 'a_b3')]):
        arr(ax, x, 3.0, x, 2.88, color='#999999', lw=1.0)
        rbox(ax, x, 2.20, 2.55, 1.30, T(k), fc, fs=7.0)

    ax.text(6.6, 0.70, T('a_foot'), ha='center', fontsize=7.2,
            style='italic', color='#555555')
    save(fig, 'figA_physics', lang); plt.close(fig)


# ------------------------------------------------------------------ 图B
def fig_b(lang):
    setfont(lang)
    T = lambda k: TXT[k][lang]
    fig, ax = plt.subplots(figsize=(7.16, 4.6))
    ax.set_xlim(-3.3, 12.9); ax.set_ylim(0, 10.2); ax.axis('off')
    ax.text(5.3, 9.8, T('b_title'), ha='center', fontsize=10, fontweight='bold')

    for y, k in ((8.6, 'b_layer_data'), (7.05, 'b_layer_ev'),
                 (5.3, 'b_layer_dec'), (3.5, 'b_layer_eval')):
        ax.text(-3.2, y, T(k), ha='left', va='center', fontsize=8, color='#333333')

    # data layer
    rbox(ax, 1.0, 8.6, 2.5, 1.0, T('b_d1'), '#eeeeee', fs=7.0)
    rbox(ax, 4.3, 8.6, 3.4, 1.0, T('b_d2'), '#eeeeee', fs=7.0)
    rbox(ax, 8.8, 8.6, 4.4, 1.0, T('b_d3'), '#eeeeee', fs=7.0)
    arr(ax, 2.4, 8.6, 2.8, 8.6); arr(ax, 6.1, 8.6, 6.5, 8.6)

    # evidence layer
    ax.text(-3.2, 6.35, T('b_e_sub'), ha='left', va='center', fontsize=7,
            color='#2e7d32', linespacing=1.3)
    for x, k, ls, fc in ((1.0, 'b_e1', '-', '#d8f0d8'), (4.3, 'b_e2', '-', '#d8f0d8'),
                         (7.6, 'b_e3', '--', '#f0f0f0'), (10.9, 'b_e4', '--', '#f0f0f0')):
        rbox(ax, x, 7.05, 3.0, 1.0, T(k), fc, fs=7.0, ls=ls)

    # decision layer
    for x, k, ls, fc, lw in ((1.0, 'b_c1', '-', '#fdf3cf', 1.1),
                             (4.3, 'b_c2', '-', '#fdf3cf', 1.1),
                             (7.6, 'b_c3', '-', '#fbe9e7', 1.8),
                             (10.9, 'b_c4', '--', '#f0f0f0', 1.1)):
        rbox(ax, x, 5.3, 3.0, 1.1, T(k), fc, fs=7.0, ls=ls, lw=lw)

    # evaluation layer
    rbox(ax, 1.0, 3.5, 3.0, 1.1, T('b_v1'), '#d6e6f7', fs=6.8)
    rbox(ax, 4.3, 3.5, 3.0, 1.1, T('b_v2'), '#d6e6f7', fs=6.8)
    rbox(ax, 7.6, 3.5, 3.0, 1.1, T('b_v3'), '#d6e6f7', fs=6.8)
    rbox(ax, 10.2, 1.9, 5.2, 1.4, T('b_v4'), '#ffffff', fs=6.2, ls='--')

    for x in (1.0, 4.3, 7.6, 10.9):
        arr(ax, x, 8.05, x, 7.6); arr(ax, x, 6.5, x, 5.9); arr(ax, x, 4.7, x, 4.1)
    arr(ax, 10.2, 2.6, 8.9, 2.95, color='#999999', ls='--')

    ax.text(5.3, 0.8, T('b_foot'), ha='center', fontsize=8, color='#333333')
    save(fig, 'figB_architecture', lang); plt.close(fig)


# ------------------------------------------------------------------ 图C
def fig_c(lang):
    setfont(lang)
    T = lambda k: TXT[k][lang]
    fig, ax = plt.subplots(figsize=(7.16, 3.8))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.4); ax.axis('off')
    ax.text(7.4, 5.95, T('c_title'), ha='center', fontsize=10, fontweight='bold')

    xs = [1.1, 3.2, 5.3, 7.4, 9.5, 11.6, 13.7]
    y = 4.5
    rbox(ax, xs[0], y, 1.95, 1.0, T('c1'), '#eeeeee', fs=6.4)
    rbox(ax, xs[1], y, 1.95, 1.0, T('c2'), '#eeeeee', fs=6.4)
    rbox(ax, xs[2], y + 0.55, 1.95, 0.85, T('c3'), '#d8f0d8', fs=6.4)
    rbox(ax, xs[2], y - 0.55, 1.95, 0.85, T('c4'), '#d8f0d8', fs=6.4)
    rbox(ax, xs[3], y, 1.95, 1.0, T('c5'), '#fbe9e7', fs=6.4)
    rbox(ax, xs[4], y, 1.95, 1.0, T('c6'), '#d6e6f7', fs=6.4)
    rbox(ax, xs[5], y, 1.95, 1.0, T('c7'), '#d6e6f7', fs=6.4)
    rbox(ax, xs[6], y, 1.7, 1.0, T('c8'), '#fdf3cf', fs=6.4)

    arr(ax, 2.08, y, 2.22, y)
    arr(ax, 4.18, y, 4.32, y + 0.52); arr(ax, 4.18, y, 4.32, y - 0.52)
    arr(ax, 6.28, y + 0.55, 6.42, y + 0.28); arr(ax, 6.28, y - 0.55, 6.42, y - 0.28)
    arr(ax, 8.38, y, 8.52, y); arr(ax, 10.48, y, 10.62, y); arr(ax, 12.58, y, 12.72, y)

    for x, sx, k in ((6.6, 7.4, 'c_n1'), (9.5, 9.5, 'c_n2'), (12.4, 11.6, 'c_n3')):
        arr(ax, sx, 3.85, x, 3.35, color='#999999', ls='--')
        rbox(ax, x, 2.5, 2.5, 1.45, T(k), '#f7f7f7', fs=6.4, ls='--', ec='#999999')

    ax.text(7.5, 0.8, T('c_foot'), ha='center', fontsize=7.2,
            style='italic', color='#555555')
    save(fig, 'figC_configF_flow', lang); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    # 中文版已按要求移除；如需中文版，把 'zh' 加回下面的循环即可。
    for _lang in ('en',):
        fig_a(_lang); fig_b(_lang); fig_c(_lang)
    print('all figures done ->', OUT)
