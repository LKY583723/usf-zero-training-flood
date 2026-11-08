import os
import numpy as np
import rasterio

root = 'D:/DisasterM3/UrbanSARFloods/urban_sar_floods'


def parse_split(root, split='train'):
    txt = os.path.join(root, 'Valid_dataset.txt' if split == 'valid' else 'Train_dataset.txt')
    samples = []
    with open(txt, encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        gt_rel = ln.split()[0]
        gt_rel = gt_rel.replace('..', '').strip('/\\')
        gt_path = os.path.normpath(os.path.join(root, gt_rel))
        if not os.path.exists(gt_path):
            continue
        d, fn = os.path.split(gt_rel)
        d_sar = os.path.normpath(os.path.join(os.path.dirname(d), 'SAR'))
        fn_sar = fn.replace('_GT.tif', '_SAR.tif')
        sar_path = os.path.normpath(os.path.join(root, d_sar, fn_sar))
        if os.path.exists(sar_path):
            samples.append((sar_path, gt_path))
    return samples


train = parse_split(root, 'train')
valid = parse_split(root, 'valid')
print('train:', len(train), 'valid:', len(valid))
print('示例:', os.path.basename(train[0][0]), '<-', os.path.basename(train[0][1]))

with rasterio.open(train[0][0]) as ds:
    sar = ds.read()
    print('SAR shape:', sar.shape, 'dtype:', sar.dtype)
with rasterio.open(train[0][1]) as ds:
    gt = ds.read(1)
    print('GT shape:', gt.shape, '类:', np.unique(gt))
