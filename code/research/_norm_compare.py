"""快速对比 z-score vs min-max 归一化（5 epochs，子集），供重训决策。"""
import os
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from data_loader import CLASS_NUM, UrbanSARFloodDataset
import segmentation_models_pytorch as smp

norm = os.environ.get('USF_NORM', 'zscore')
root = 'D:/DisasterM3/UrbanSARFloods/urban_sar_floods'
print(f'=== NORM={norm} ===', flush=True)

train_ds = UrbanSARFloodDataset(root, 'train', crop=256, augment=True, preload=True)
valid_ds = UrbanSARFloodDataset(root, 'valid', crop=256, augment=False, preload=True)
train_sub = Subset(train_ds, range(300))
valid_sub = Subset(valid_ds, range(150))
print(f'子集 train={len(train_sub)} valid={len(valid_sub)}', flush=True)

model = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                 in_channels=8, classes=3).cuda()
w = 1.0 / np.maximum(CLASS_NUM, 1.0); w = w / w.mean()
crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).cuda())
opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

train_loader = DataLoader(train_sub, batch_size=12, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_sub, batch_size=12, num_workers=0)

for ep in range(1, 6):
    model.train()
    tot = 0
    for sar, gt in train_loader:
        sar, gt = sar.cuda(), gt.cuda()
        loss = crit(model(sar), gt)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    # valid open IoU
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for sar, gt in valid_loader:
            logits = model(sar.cuda())
            pred = logits.argmax(1).cpu().numpy()
            gt = gt.numpy()
            p = pred == 1; t = gt == 1
            tp += (p & t).sum(); fp += (p & ~t).sum(); fn += (~p & t).sum()
    oiou = tp / max(tp + fp + fn, 1)
    print(f'  ep{ep}: loss={tot/len(train_loader):.4f} valid_open_IoU={oiou:.4f}', flush=True)
print(f'=== NORM={norm} 完成 ===', flush=True)
