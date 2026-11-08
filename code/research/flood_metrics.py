#!/usr/bin/env python3
"""GE 洪水制图 · 像素级评测（复用论文表二混淆矩阵口径）。

真值 = UrbanSARFloods 真实 GT（0=无洪水, 1=开阔洪水, 2=城市洪水/水）。
对每个 tile 算 IoU/F1/Kappa/Precision/Recall/Specificity，
并按 GT 类别分解（开阔=1、城区=2），聚合成事件级/总体。
"""
import numpy as np


def confusion(pred, gt_binary):
    tp = int(np.logical_and(pred, gt_binary).sum())
    fp = int(np.logical_and(pred, ~gt_binary).sum())
    fn = int(np.logical_and(~pred, gt_binary).sum())
    tn = int(np.logical_and(~pred, ~gt_binary).sum())
    return tp, fp, fn, tn


def metrics(pred, gt_binary):
    """pred/gt_binary 同形状 bool。返回全指标 dict。"""
    tp, fp, fn, tn = confusion(pred, gt_binary)
    n = tp + fp + fn + tn
    if n == 0:
        return None
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    oa = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (oa - pe) / (1 - pe) if (1 - pe) else 0.0
    return {"iou": iou, "f1": f1, "kappa": kappa, "precision": prec,
            "recall": rec, "specificity": spec, "oa": oa}


def decompose_by_class(pred, gt):
    """按 GT 类别分解指标。
    gt: 0/1/2。返回 {"open": metrics, "urban": metrics, "all": metrics}。
    """
    out = {}
    g1 = gt == 1  # 开阔洪水
    g2 = gt == 2  # 城市洪水/水
    gany = g1 | g2
    out["open"] = metrics(pred, g1)
    out["urban"] = metrics(pred, g2)
    out["all"] = metrics(pred, gany)
    return out


def aggregate(tile_metrics, keys=("iou", "f1", "kappa", "precision", "recall")):
    """tile_metrics: list of (class, metrics dict)。按 class 聚合均值。"""
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for cls, m in tile_metrics:
        if m is None:
            continue
        for k in keys:
            agg[cls][k].append(m[k])
    out = {}
    for cls, d in agg.items():
        out[cls] = {k: float(np.mean(v)) for k, v in d.items()}
    return out
