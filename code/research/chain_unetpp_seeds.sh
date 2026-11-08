#!/bin/bash
# 链路：
#   1) 等 unet zscore 训练完成（grep train_unet_zscore.log 的 [DONE]）
#   2) zscore 评测 unet zscore
#   3) 训练 unet++ seed0（zscore）→ 评测
#   4) 训练 unet++ seed123（zscore）→ 评测
REPRO="D:/DisasterM3/UrbanSARFloods/repro"
UZ_LOG="$REPRO/train_unet_zscore.log"
OUT="$REPRO/ckpts_zscore"

echo "=== [chain] 等待 unet zscore 训练完成 ==="
for i in $(seq 1 200); do
  if grep -q "\[DONE\]" "$UZ_LOG" 2>/dev/null; then
    echo "=== [chain] unet zscore DONE ==="
    break
  fi
  sleep 30
done
if ! grep -q "\[DONE\]" "$UZ_LOG" 2>/dev/null; then
  echo "=== [chain] 超时：unet zscore 未在 100min 内完成，终止 ==="
  exit 1
fi

cd "$REPRO"

echo "=== [chain] 评测 unet zscore ==="
python _eval_baselines.py unet "$OUT/unet_best_unet_zscore.pth"

echo "=== [chain] 训练 unet++ seed0（zscore）==="
python -u train.py --data-root D:/DisasterM3/UrbanSARFloods/urban_sar_floods \
  --out "$OUT" --epochs 100 --arch unet++ --tag unetpp_seed0 --preload train --seed 0 \
  2>&1 | tee train_unetpp_seed0.log

echo "=== [chain] 评测 unet++ seed0 ==="
python _eval_baselines.py unet++ "$OUT/unet++_best_unetpp_seed0.pth"

echo "=== [chain] 训练 unet++ seed123（zscore）==="
python -u train.py --data-root D:/DisasterM3/UrbanSARFloods/urban_sar_floods \
  --out "$OUT" --epochs 100 --arch unet++ --tag unetpp_seed123 --preload train --seed 123 \
  2>&1 | tee train_unetpp_seed123.log

echo "=== [chain] 评测 unet++ seed123 ==="
python _eval_baselines.py unet++ "$OUT/unet++_best_unetpp_seed123.pth"

echo "=== [chain] 全部完成 ==="
