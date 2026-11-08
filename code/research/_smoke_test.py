import sys
sys.path.insert(0, 'D:/DisasterM3/UrbanSARFloods/repro')
import torch
import numpy as np
from data_loader import UrbanSARFloodDataset
import segmentation_models_pytorch as smp

ds = UrbanSARFloodDataset('D:/DisasterM3/UrbanSARFloods/urban_sar_floods', 'train',
                          crop=256, augment=True)
print('dataset len:', len(ds))
sar, gt = ds[0]
print('sample SAR:', tuple(sar.shape), 'GT:', tuple(gt.shape), '类:', np.unique(gt.numpy()))

model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=8, classes=3)
model = model.cuda()
print('参数量: %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

opt = torch.optim.Adam(model.parameters(), lr=1e-5)
crit = torch.nn.CrossEntropyLoss()
batch = torch.stack([ds[i][0] for i in range(2)]).cuda()
gtb = torch.stack([ds[i][1] for i in range(2)]).cuda()
print('batch:', batch.shape, 'gt:', gtb.shape)

out = model(batch)
print('logits:', tuple(out.shape))
loss = crit(out, gtb)
loss.backward()
opt.step()
print('SMOKE OK, loss=%.4f' % loss.item())
