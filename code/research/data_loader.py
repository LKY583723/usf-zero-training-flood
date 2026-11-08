#!/usr/bin/env python3
"""UrbanSARFloods 复现 · 数据加载器（SMP 训练用）。

读取 urban_sar_floods/ 训练包（作者发布结构）：
  urban_sar_floods/
    01_NF/ 02_FO/ 03_FU/          # 无洪水 / 开放洪水 / 城市洪水 tile 类别
      GT/*.tif                    # 单波段，0/1/2（1=开放洪水, 2=城市洪水）
      SAR/*.tif                   # 8 波段 float32（band0-3 相干度, band4-7 强度 dB）
    Train_dataset.txt             # 训练样本列表
    Valid_dataset.txt             # 验证样本列表

归一化：相干度 0~1 直接用；强度 dB clip[-30,0] → [0,1]。
训练增强：512→256 随机裁剪 + 水平/垂直翻转 + 90/180/270 旋转（论文配置）。
"""
import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# 波段语义（与 flood_strategies.py 一致）
CO_PRE_VH, CO_PRE_VV, CO_CO_VH, CO_CO_VV = 0, 1, 2, 3
INT_PRE_VH, INT_PRE_VV, INT_POST_VH, INT_POST_VV = 4, 5, 6, 7
DB_LO, DB_HI = -30.0, 0.0


def parse_split(root, split="train"):
    """解析 Train/Valid_dataset.txt → [(sar_path, gt_path), ...]。

    txt 每行是 GT 相对路径（如 '../03_FU/GT/xxx_GT.tif'），
    SAR 由 GT 路径替换 GT→SAR、_GT→_SAR 得到。
    """
    txt = os.path.join(root, "Valid_dataset.txt" if split == "valid" else "Train_dataset.txt")
    samples = []
    if os.path.exists(txt):
        with open(txt, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for ln in lines:
            gt_rel = ln.split()[0].lstrip("./\\")
            gt_path = os.path.normpath(os.path.join(root, gt_rel))
            if not os.path.exists(gt_path):
                continue
            # 目录 GT→SAR，文件名 _GT→_SAR
            d, fn = os.path.split(gt_rel)
            d_sar = os.path.normpath(os.path.join(os.path.dirname(d), "SAR"))
            fn_sar = fn.replace("_GT.tif", "_SAR.tif")
            sar_path = os.path.normpath(os.path.join(root, d_sar, fn_sar))
            if os.path.exists(sar_path):
                samples.append((sar_path, gt_path))
        if samples:
            return samples
    # 兜底：扫描类别目录
    for cat in ("01_NF", "02_FO", "03_FU"):
        d = os.path.join(root, cat)
        sar_dir = os.path.join(d, "SAR")
        if not os.path.isdir(sar_dir):
            continue
        for fn in sorted(os.listdir(sar_dir)):
            if not fn.endswith(".tif"):
                continue
            sar = os.path.join(sar_dir, fn)
            gt = os.path.join(d, "GT", fn)
            if os.path.exists(gt):
                samples.append((sar, gt))
    return samples


# 作者 data_norm.txt 提供的逐通道归一化参数（z-score + min-max）
MEAN_IN = np.array([0.23651549, 0.31761484, 0.18514981, 0.26901252,
                    -14.57879175, -8.6098158, -14.29073382, -8.33534564],
                   dtype=np.float32)
STD_IN = np.array([0.16280619, 0.20849304, 0.14008107, 0.19767644,
                   4.07141682, 3.94773216, 4.21006244, 4.05494136],
                  dtype=np.float32)
MIN_IN = np.array([0.03540783, 0.04156244, 0.03062815, 0.03643807,
                   -23.19632721, -16.43351936, -23.46723938, -16.71599197],
                  dtype=np.float32)
MAX_IN = np.array([0.56519184, 0.70935154, 0.53432751, 0.66288453,
                   -7.85489845, -1.61887217, -7.48478746, -1.26720715],
                  dtype=np.float32)
# 作者统计的类别像素数（WCE 权重）
CLASS_NUM = np.array([1161425968, 18120626, 363550], dtype=np.float64)

# 归一化方式：zscore | minmax
NORM_MODE = os.environ.get("USF_NORM", "zscore")


def normalize(sar):
    """8 波段归一化（作者 data_norm.txt 参数，zscore 或 minmax）。"""
    sar = np.nan_to_num(sar, nan=0.0).astype(np.float32)
    if NORM_MODE == "minmax":
        return (sar - MIN_IN[:, None, None]) / (MAX_IN[:, None, None] - MIN_IN[:, None, None])
    return (sar - MEAN_IN[:, None, None]) / STD_IN[:, None, None]


def random_crop_augment(sar, gt, crop=256, rng=None):
    """随机裁剪到 crop×crop + 随机翻转/旋转（论文增强）。"""
    if rng is None:
        rng = np.random
    h, w = sar.shape[1], sar.shape[2]
    y0 = rng.randint(0, h - crop + 1) if h > crop else 0
    x0 = rng.randint(0, w - crop + 1) if w > crop else 0
    sar = sar[:, y0:y0 + crop, x0:x0 + crop]
    gt = gt[y0:y0 + crop, x0:x0 + crop]
    if rng.rand() < 0.5:
        sar, gt = sar[:, :, ::-1], gt[:, ::-1]
    if rng.rand() < 0.5:
        sar, gt = sar[:, ::-1, :], gt[::-1, :]
    k = int(rng.randint(0, 4))
    if k:
        sar = np.rot90(sar, k, axes=(1, 2))
        gt = np.rot90(gt, k)
    return sar.copy(), gt.copy()


class UrbanSARFloodDataset(Dataset):
    def __init__(self, root, split="train", crop=256, augment=True, preload=False):
        self.samples = parse_split(root, split)
        self.crop = crop
        self.augment = augment and split == "train"
        self._mem = None
        if preload:
            self._preload()

    def __len__(self):
        return len(self.samples)

    def _preload(self):
        """一次性读全部样本到内存（SAR float16 + GT int8）。
        512×512：SAR 4MB/样本 + GT 0.25MB/样本；4501 样本 ≈ 19GB。"""
        sars, gts = [], []
        for i, (sar_p, gt_p) in enumerate(self.samples):
            with rasterio.open(sar_p) as ds:
                sars.append(ds.read().astype(np.float16))
            with rasterio.open(gt_p) as ds:
                gts.append(ds.read(1).astype(np.int8))
            if (i + 1) % 500 == 0:
                print(f"  preload {i + 1}/{len(self.samples)}", flush=True)
        self._mem = (sars, gts)
        print(f"[preload] {len(self.samples)} 样本已载入内存")

    def _raw(self, idx):
        if self._mem is not None:
            return self._mem[0][idx], self._mem[1][idx]
        sar_p, gt_p = self.samples[idx]
        with rasterio.open(sar_p) as ds:
            sar = ds.read().astype(np.float16)
        with rasterio.open(gt_p) as ds:
            gt = ds.read(1).astype(np.int8)
        return sar, gt

    def __getitem__(self, idx):
        sar, gt = self._raw(idx)
        if self.augment:
            sar, gt = random_crop_augment(sar, gt, self.crop)
        else:
            h, w = sar.shape[1], sar.shape[2]
            if h != self.crop or w != self.crop:
                y0, x0 = h // 2, w // 2
                sar = sar[:, y0 - self.crop // 2:y0 + self.crop // 2,
                          x0 - self.crop // 2:x0 + self.crop // 2]
                gt = gt[y0 - self.crop // 2:y0 + self.crop // 2,
                        x0 - self.crop // 2:x0 + self.crop // 2]
        sar = normalize(sar)  # 裁剪后逐通道 z-score
        return torch.from_numpy(sar), torch.from_numpy(gt).long()
