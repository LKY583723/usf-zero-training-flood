#!/bin/bash
# 统一 zscore 基线链路：
#   1) 等 deeplabv3+ 训练完成（grep train_deeplabv3p.log 的 [DONE]）
#   2) zscore 评测 deeplabv3+
#   3) 训练 unet zscore（batch12, 默认 zscore, seed42）
#   4) zscore 评测 unet
REPRO="D:/DisasterM3/UrbanSARFloods/repro"
DL_LOG="$REPRO/train_deeplabv3p.log"

echo "=== [chain] 等待 deeplabv3+ 训练完成 ==="
for i in $(seq 1 300); do
  if grep -q "\[DONE\]" "$DL_LOG" 2>/dev/null; then
    echo "=== [chain] deeplabv3+ DONE ==="
    break
  fi
  sleep 30
done
if ! grep -q "\[DONE\]" "$DL_LOG" 2>/dev/null; then
  echo "=== [chain] 超时：deeplabv3+ 未在 150min 内完成，终止 ==="
  exit 1
fi

cd "$REPRO"

echo "=== [chain] 评测 deeplabv3+（zscore）==="
python _eval_baselines.py "deeplabv3+" "ckpts_minmax/deeplabv3+_best_deeplabv3p.pth"

echo "=== [chain] 训练 unet zscore（batch12, 默认 zscore, seed42）==="
python -u train.py --data-root D:/DisasterM3/UrbanSARFloods/urban_sar_floods \
  --out "$REPRO/ckpts_zscore" --epochs 100 --arch unet --tag unet_zscore --preload train \
  2>&1 | tee train_unet_zscore.log

echo "=== [chain] 评测 unet zscore ==="
python _eval_baselines.py unet "$REPRO/ckpts_zscore/unet_best_unet_zscore.pth"

echo "=== [chain] 全部完成 ==="
