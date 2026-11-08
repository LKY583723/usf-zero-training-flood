#!/usr/bin/env python3
"""UrbanSARFloods 复现 · SMP 训练（复刻论文 baseline 配置）。

配置（论文提取）：
  模型     Unet (encoder resnet34, in_channels=8, classes=3, 从头训练)
  损失     Weighted Cross-Entropy（权重按类频率反比）
  优化     Adam, lr=1e-5, weight_decay=1e-4
  批次     batch=12（8GB 显存自动降 batch + 梯度累积补到 12）
  轮数     100 epochs
  增强     512→256 随机裁剪 + 翻转/旋转

用法：
  python train.py --data-root D:/DisasterM3/UrbanSARFloods/urban_sar_floods \
      --out D:/DisasterM3/UrbanSARFloods/repro/ckpts --epochs 100
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import CLASS_NUM, UrbanSARFloodDataset

MODELS = {
    "unet": lambda ch, cl: __import__("segmentation_models_pytorch").Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=ch, classes=cl),
    "unet++": lambda ch, cl: __import__("segmentation_models_pytorch").UnetPlusPlus(
        encoder_name="resnet34", encoder_weights=None, in_channels=ch, classes=cl),
    "fpn": lambda ch, cl: __import__("segmentation_models_pytorch").FPN(
        encoder_name="resnet34", encoder_weights=None, in_channels=ch, classes=cl),
    "pspnet": lambda ch, cl: __import__("segmentation_models_pytorch").PSPNet(
        encoder_name="resnet34", encoder_weights=None, in_channels=ch, classes=cl),
    "manet": lambda ch, cl: __import__("segmentation_models_pytorch").MAnet(
        encoder_name="resnet34", encoder_weights=None, in_channels=ch, classes=cl),
    "deeplabv3+": lambda ch, cl: __import__("segmentation_models_pytorch").DeepLabV3Plus(
        encoder_name="resnet34", encoder_weights=None, in_channels=ch, classes=cl),
}


def class_freq_weights(n_cls=3):
    """WCE 权重 = 类别像素数反比（用作者 data_norm.txt 的 CLASS_NUM，全数据集统计）。"""
    w = 1.0 / np.maximum(CLASS_NUM, 1.0)
    w = w / w.mean()
    return torch.tensor(w[:n_cls], dtype=torch.float32)


def iou_per_class(pred, target, n_cls=3):
    """pred/target: (N,H,W) 类别索引。返回每类 IoU。"""
    ious = []
    for c in range(n_cls):
        p = pred == c
        t = target == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        ious.append(inter / union if union else float("nan"))
    return ious


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = np.zeros(3); fp = np.zeros(3); fn = np.zeros(3)
    for sar, gt in loader:
        sar = sar.to(device)
        logits = model(sar)
        pred = logits.argmax(1).cpu().numpy()
        gt = gt.numpy()
        for c in range(3):
            p = pred == c; t = gt == c
            tp[c] += (p & t).sum(); fp[c] += (p & ~t).sum(); fn[c] += (~p & t).sum()
    ious = tp / np.maximum(tp + fp + fn, 1)
    return ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="ckpts")
    ap.add_argument("--arch", default="unet", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=12, help="论文 batch=12")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=1, help="batch=12 无需累积")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true",
                    help="开启 torch 确定性算法（cudnn.deterministic + use_deterministic_algorithms）。"
                         "注意：论文报告数字来自未开确定性的标准运行；开此开关后 F1 可能在小范围内（seed 方差带内）偏移。")
    ap.add_argument("--tag", default="", help="输出文件后缀：unet_best_<tag>.pth（多 seed 用 seedN）")
    ap.add_argument("--preload", default="train", choices=["none", "train", "all"],
                    help="预加载: train=仅训练集(~19GB,推荐); all=训练+验证(~27GB,原配置,32GB内存有风险); none=全部磁盘读取")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[env] 已开启确定性算法（cudnn.deterministic + use_deterministic_algorithms）")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} gpu={torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}")
    if device == "cpu":
        print("[WARN] 无 CUDA，训练将非常慢")

    os.makedirs(args.out, exist_ok=True)
    train_ds = UrbanSARFloodDataset(args.data_root, "train", args.crop,
                                    augment=True, preload=args.preload in ("train", "all"))
    valid_ds = UrbanSARFloodDataset(args.data_root, "valid", args.crop,
                                    augment=False, preload=args.preload == "all")
    print(f"[data] train={len(train_ds)} valid={len(valid_ds)} preload={args.preload}")

    weights = class_freq_weights()
    print(f"[wce] 类权重 (NF/FO/FU): {weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = MODELS[args.arch](8, 3).to(device)
    print(f"[model] {args.arch} 参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    log = {"epochs": [], "train_loss": [], "valid_iou": []}
    best_iou, best_ep = -1, -1
    acc_steps = max(1, args.grad_accum)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, nb = 0.0, 0
        opt.zero_grad()
        for i, (sar, gt) in enumerate(train_loader):
            sar, gt = sar.to(device), gt.to(device)
            loss = criterion(model(sar), gt) / acc_steps
            loss.backward()
            if (i + 1) % acc_steps == 0:
                opt.step(); opt.zero_grad()
            tot_loss += loss.item() * acc_steps; nb += 1
        opt.step(); opt.zero_grad()

        ious = evaluate(model, valid_loader, device)
        miou = np.nanmean(ious)
        log["epochs"].append(ep)
        log["train_loss"].append(tot_loss / max(nb, 1))
        log["valid_iou"].append(ious.tolist())
        print(f"[ep {ep}/{args.epochs}] loss={tot_loss/max(nb,1):.4f} "
              f"val_iou(open/urban/all)={ious[1]:.4f}/{ious[2]:.4f}/{ious[0]:.4f} "
              f"miou={miou:.4f} ({time.time()-t0:.0f}s)")
        if miou > best_iou:
            best_iou, best_ep = miou, ep
            torch.save(model.state_dict(), os.path.join(args.out, f"{args.arch}_best_{args.tag}.pth" if args.tag else f"{args.arch}_best.pth"))
            print(f"  → best miou {miou:.4f} 已保存")

    log_name = f"train_log_{args.tag}.json" if args.tag else "train_log.json"
    with open(os.path.join(args.out, log_name), "w") as f:
        json.dump({"args": vars(args), "best_ep": best_ep, "best_miou": best_iou,
                   "log": log}, f, indent=2)
    print(f"[DONE] best miou={best_iou:.4f} @ep{best_ep} → {os.path.join(args.out, log_name)}")


if __name__ == "__main__":
    main()
