#!/usr/bin/env python3
"""UrbanSARFloods 复现 · 测试集评测（整景滑窗推理）。

用法：
  python test.py --ckpt D:/DisasterM3/UrbanSARFloods/repro/ckpts/unet_best.pth \
      --data-root D:/DisasterM3/UrbanSARFloods \
      --events 20210727_Weihui 20230609_NovaKakhovka

输出：results/flood_repro_test.md（按 open/urban 报 F1/IoU，对比论文 0.77/<0.1）
"""
import argparse
import json
import os

import numpy as np
import rasterio
import torch

from data_loader import normalize


def confusion(pred, gt_binary):
    tp = int(np.logical_and(pred, gt_binary).sum())
    fp = int(np.logical_and(pred, ~gt_binary).sum())
    fn = int(np.logical_and(~pred, gt_binary).sum())
    tn = int(np.logical_and(~pred, ~gt_binary).sum())
    return tp, fp, fn, tn


def metrics(pred, gt_binary):
    tp, fp, fn, tn = confusion(pred, gt_binary)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "iou": iou, "f1": f1}


@torch.no_grad()
def predict_scene(model, sar_path, device, win=256, step=256):
    """整景 256² 非重叠网格推理 → 3 类掩膜（对齐论文 patch 口径）。"""
    with rasterio.open(sar_path) as ds:
        H, W = ds.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        count = np.zeros((H, W), dtype=np.uint8)
        for y0 in range(0, H - win + 1, step):
            for x0 in range(0, W - win + 1, step):
                sar = ds.read(window=((y0, y0 + win), (x0, x0 + win))).astype(np.float32)
                sar = normalize(sar)
                inp = torch.from_numpy(sar).unsqueeze(0).to(device)
                logits = model(inp)
                pred = logits.argmax(1).squeeze(0).cpu().numpy()
                mask[y0:y0 + win, x0:x0 + win] += pred.astype(np.uint8)
                count[y0:y0 + win, x0:x0 + win] += 1
        count = np.maximum(count, 1)
        mask = np.round(mask.astype(np.float32) / count).astype(np.uint8)
    return mask


def eval_event(model, event, data_root, device, patch=256):
    """论文同口径：256² 网格，只评测 GT 有洪水的 patch，patch 级 F1 平均。"""
    sub = os.path.join(event, "testing_case_orig", event)
    sar_path = os.path.join(data_root, sub, f"{event}_SAR.tif")
    if not os.path.exists(sar_path):  # Jubba 共享 SAR 文件名特例
        sar_path = os.path.join(data_root, sub, "20231201_Jubba_SAR.tif")
    gt_path = os.path.join(data_root, sub, f"{event}_GT.tif")
    print(f"[{event}] 256² 网格推理 {os.path.basename(sar_path)} ...", flush=True)
    pred = predict_scene(model, sar_path, device, win=patch, step=patch)
    with rasterio.open(gt_path) as ds:
        gt = ds.read(1)
    H, W = gt.shape

    # 逐 patch：只统计 GT 洪水像素 ≥ 100 的 patch；像素聚合（对齐论文 class-level F1）
    acc = {"open": [0, 0, 0], "urban": [0, 0, 0], "all": [0, 0, 0]}  # tp, fp, fn
    n_patch = 0
    for y0 in range(0, H - patch + 1, patch):
        for x0 in range(0, W - patch + 1, patch):
            g = gt[y0:y0 + patch, x0:x0 + patch]
            if (g > 0).sum() < 100:
                continue
            p = pred[y0:y0 + patch, x0:x0 + patch]
            n_patch += 1
            for cls, gbin in (("open", g == 1), ("urban", g == 2), ("all", g > 0)):
                tp, fp, fn, _ = confusion(p > 0, gbin)
                acc[cls][0] += tp; acc[cls][1] += fp; acc[cls][2] += fn

    out = {}
    for cls, (tp, fp, fn) in acc.items():
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        out[cls] = {"f1": f1, "iou": iou, "precision": p_, "recall": r_}
    out["n_flood_patch"] = n_patch
    print(f"   洪水 patch: {n_patch} | open F1={out['open']['f1']:.3f} "
          f"urban F1={out['urban']['f1']:.3f} all IoU={out['all']['iou']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="D:/DisasterM3/UrbanSARFloods")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--events", nargs="*", default=["20210727_Weihui", "20230609_NovaKakhovka"])
    ap.add_argument("--out", default="D:/DisasterM3/experiment1/results/flood_repro_test.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=8, classes=3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    results = {}
    for ev in args.events:
        try:
            results[ev] = eval_event(model, ev, args.data_root, device)
        except Exception as e:
            print(f"[skip] {ev}: {type(e).__name__} {str(e)[:120]}")
            results[ev] = None

    lines = ["# UrbanSARFloods 复现 · 测试集评测（整景）", "",
             f"模型: Unet(resnet34, 8ch) 从头训练 | 对比论文: 开放 F1 0.77 / 城市 <0.1", "",
             "| 事件 | 范围 | F1 | IoU | Precision | Recall |",
             "|:--|:--|:--|:--|:--|:--|"]
    for ev, r in results.items():
        if r is None:
            lines.append(f"| {ev} | - | 失败 | - | - | - |")
            continue
        for label, key in (("总体", "all"), ("开阔", "open"), ("城区", "urban")):
            m = r[key]
            lines.append(f"| {ev} | {label} | {m['f1']:.3f} | {m['iou']:.3f} | "
                         f"{m['precision']:.3f} | {m['recall']:.3f} |")
    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(os.path.dirname(args.out), "flood_repro_test.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(md)
    print(f"已保存 → {args.out}")


if __name__ == "__main__":
    main()
