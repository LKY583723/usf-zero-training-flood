#!/usr/bin/env python3
"""
GE 洪水制图 · 确定性候选洪水掩膜策略。

输入：UrbanSARFloods 8 波段 SAR 块（float32，dB/相干度）：
  band0,1 = 灾前相干度 (VH, VV)     —— 灾前参照（2026-08-28 审计修正：原注释误写为"跨灾"，实测测试 patch band0,1>band2,3，灾前相干度更高）
  band2,3 = 跨灾（洪水期）相干度 (VH, VV) —— 被淹区去相干
  band4,5 = 灾前强度 dB (VH, VV)
  band6,7 = 灾后强度 dB (VH, VV)

三种零训练策略（Otsu 自动阈值，与论文表二同口径）：
  S_int  灾后强度暗水阈值      —— 开阔区洪水（水在 SAR 上暗）
  S_chg  |灾后−灾前| 强度变化   —— 新增暗区（淹没前本是陆地）
  S_coh  相干变化 Δγ=pre−co    —— 去相干区（城区洪水关键）

输出：三个二值掩膜（True=预测洪水）。
"""
import cv2
import numpy as np

# 波段索引（用 GT 洪水区验证：band0,1=灾前相干度，band2,3=跨灾/洪水期相干度）
CO_PRE_VH, CO_PRE_VV = 0, 1     # 灾前相干度
CO_CO_VH, CO_CO_VV = 2, 3       # 跨灾（洪水期）相干度
INT_PRE_VH, INT_PRE_VV = 4, 5
INT_POST_VH, INT_POST_VV = 6, 7

DB_LO, DB_HI = -30.0, 0.0     # 强度 dB 拉伸范围


def _finite(img):
    return np.nan_to_num(img, nan=np.nanmedian(img))


def _otsu_binary(values, flip=False):
    """对有效像素做 Otsu，返回二值掩膜。
    flip=False：> 阈值 = 正类；flip=True：< 阈值 = 正类。
    nan/无效像素永远不为正类。"""
    v = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(v)
    finite = v[valid]
    if len(finite) < 100:
        return np.zeros(v.shape, dtype=bool)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        return np.zeros(v.shape, dtype=bool)
    q = np.clip((v - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    ret, _ = cv2.threshold(q, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pred = (q < ret) if flip else (q > ret)
    return pred & valid


def _db_to_uint8(db):
    return np.clip((np.nan_to_num(db, nan=DB_LO) - DB_LO) / (DB_HI - DB_LO)
                   * 255.0, 0, 255).astype(np.uint8)


def strategy_int(sar, pol="vv", otsu=True, thresh_db=None):
    """S_int：灾后强度暗水阈值。dark = 正类（flip）。"""
    post = sar[INT_POST_VV] if pol == "vv" else sar[INT_POST_VH]
    if thresh_db is not None:
        return _finite(post) < thresh_db
    return _otsu_binary(post, flip=True)


def strategy_chg(sar, pol="vv"):
    """S_chg：|灾后−灾前| 强度变化阈值（变化大 = 正类）。"""
    post = sar[INT_POST_VV] if pol == "vv" else sar[INT_POST_VH]
    pre = sar[INT_PRE_VV] if pol == "vv" else sar[INT_PRE_VH]
    diff = np.abs(_finite(post) - _finite(pre))
    return _otsu_binary(diff, flip=False)


def strategy_coh(sar, pol="vv"):
    """S_coh：相干变化 Δγ = pre_coh − co_coh 阈值（去相干大 = 正类）。"""
    co = sar[CO_CO_VV] if pol == "vv" else sar[CO_CO_VH]
    pre = sar[CO_PRE_VV] if pol == "vv" else sar[CO_PRE_VH]
    delta = _finite(pre) - _finite(co)
    return _otsu_binary(delta, flip=False)


def strategy_urban_coh(sar, pol="vh", pre_thr=0.6, delta_min=0.25, post_min=-6.0):
    """S_urban：城区洪水 —— 灾前高相干(pre_coh)区的大幅去相干(delta)。
    城区洪水：建筑区灾前高相干(建筑稳定)，被淹后显著去相干（Weihui 实测）。
    约束 pre_coh>pre_thr 排除原本低相干区假阳；post_vv>post_min 保留建筑双反弹回波。
    """
    pre = sar[CO_PRE_VH] if pol == "vh" else sar[CO_PRE_VV]
    co = sar[CO_CO_VH] if pol == "vh" else sar[CO_CO_VV]
    post = sar[INT_POST_VV] if pol == "vh" else sar[INT_POST_VV]
    delta = _finite(pre) - _finite(co)
    m = (delta > delta_min) & (pre > pre_thr) & np.isfinite(pre) & np.isfinite(co)
    if post_min is not None:
        m = m & (_finite(post) > post_min)
    return m


def combine_pol(sar, fn, mode="or"):
    """把 VV/VH 两极化结果合并：or=并集（召回优先），and=交集（精度优先）。"""
    m_vv = fn(sar, "vv")
    m_vh = fn(sar, "vh")
    if mode == "and":
        return m_vv & m_vh
    return m_vv | m_vh


def conn_filter(mask, min_size=300):
    """连通域过滤：去掉面积 < min_size 的孤立小区域（去散点假阳，保真实洪水）。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[lab == i] = True
    return out


def all_strategies(sar, int_pol="or", chg_pol="or", coh_pol="or"):
    """返回 {int, chg, coh, urban} 四个掩膜。"""
    return {
        "int": combine_pol(sar, lambda s, p: strategy_int(s, p), int_pol),
        "chg": combine_pol(sar, lambda s, p: strategy_chg(s, p), chg_pol),
        "coh": combine_pol(sar, lambda s, p: strategy_coh(s, p), coh_pol),
        "urban": combine_pol(sar, lambda s, p: strategy_urban_coh(s, p), "or"),
    }


def render_amp_rgb(sar):
    """灾后振幅 → RGB（VV→R, VH→G），供 VLM 场景分析。"""
    vv8 = _db_to_uint8(sar[INT_POST_VV])
    vh8 = _db_to_uint8(sar[INT_POST_VH])
    return np.stack([vv8, vh8, np.zeros_like(vv8)], axis=-1)


def render_coh_change_rgb(sar):
    """相干变化 Δγ → 热力图 RGB，供 VLM Verifier 查看。"""
    d = _finite(sar[CO_PRE_VV]) - _finite(sar[CO_CO_VV])
    d = np.clip(d, 0, None)
    d8 = np.clip(d / (np.percentile(d[d > 0], 95) + 1e-6) * 255, 0, 255).astype(np.uint8)
    jet = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    return cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
