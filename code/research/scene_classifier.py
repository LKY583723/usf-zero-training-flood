#!/usr/bin/env python3
"""GE 洪水制图 · 确定性场景分类（VLM 不可用时的兜底）。

用 SAR 物理特征判 scene ∈ {urban, open, vegetated, mixed}，喂 flood_router。

特征（物理依据）：
  post_vv    灾后 VV 中位强度 —— 城区建筑高回波、水体暗
  texture    灾后 VV 纹理（std）—— 城区高纹理、水体平滑
  coh_change pre_coh − co_coh（去相干量）—— 植被/水体淹没去相干
"""
import numpy as np

from flood_strategies import (CO_CO_VV, CO_PRE_VV, INT_POST_VV,
                              INT_POST_VH)  # noqa: E402


def classify_scene(sar, th_urban=-8.0, th_open=-15.0, th_coh=0.15, th_tex=8.0):
    """返回 {scene, scene_density, confidence}。sar: (8,H,W)。"""
    post_vv = np.nanmedian(sar[INT_POST_VV])
    post_vh = np.nanmedian(sar[INT_POST_VH])
    texture = np.nanstd(sar[INT_POST_VV])
    with np.errstate(invalid="ignore"):
        coh_change = np.nanmedian(sar[CO_PRE_VV] - sar[CO_CO_VV])

    # 城区：高强度（建筑强反射）
    if post_vv > th_urban:
        scene = "urban"
    # 水体/裸土：极暗
    elif post_vv < th_open:
        scene = "open"
    else:
        # 中等强度：植被淹没会显著去相干
        if coh_change > th_coh:
            scene = "vegetated"
        else:
            scene = "open" if post_vv < 0.0 else "mixed"

    density = "high" if texture > th_tex else "low"
    return {"scene": scene, "scene_density": density, "confidence": 0.7}
