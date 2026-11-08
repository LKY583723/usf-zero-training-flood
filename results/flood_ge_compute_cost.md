# 计算开销实测记录（归档 · 不进论文正文）

> 记录：2026-09-05
> **决策**：论文正文**不加精确耗时数字**——时间数字硬件相关、不可复现，会引入与核心贡献无关的审稿攻击面；且不承重（主结果 F1 不依赖它）。
> 论文已有的定性表述足够："zero training cost"、"single pre/post pair rather than a one-year GRD time series"。
> 本文件供需要时查证，不进审稿视野。

## 实测数字（口径 = 256×256 tile）

| 项 | 值 | 条件/备注 |
|---|---|---|
| **config F 单 tile 推理** | **≈21.5 ms** | Python + OpenCV 参考实现，单线程 CPU，200 个真实 Weihui patch、warmup 后取均值；**只含推理**（2×Otsu + 交集 + −12dB + 连通域），不含 I/O 与 Python 启动 |
| config F 全测试集推理（~946 待评 tile）| ≈20 s（纯计算）| 计算值，未含 I/O |
| config F 含 I/O 端到端单 tile | ≈50–70 ms（估）| GeoTIFF 读取 ~20–50ms + 推理；未实测，估算 |
| config F 训练 | **0** | 零训练、无 GPU 依赖 |
| CNN Unet++ 训练 | **6.40 h** | RTX 5060 Laptop GPU，100 epochs，~230 s/epoch（`train_unetpp.log`）|
| CNN Unet++ 单 tile 推理 | ≈7 ms | GPU（评测实测 ~152 patch/s）|

## ⚠️ 不能引用的数字

- **deeplabv3+**：训练日志被"笔记本睡眠卡死"污染（实际合计 13.4h）——不能作为典型训练时长
- **unet zscore**：`train_unet_zscore.log` 只有 51 轮计时（4.03h），外推 100 轮 ≈7.9h——日志不全，谨慎引用
- 若引用 CNN 训练时长，**只用 unet++ 的 6.40h**，并注明 GPU 型号

## 正确的叙事方式（若需要）

> 不比单 tile 速度（config F 是 CPU、CNN 是 GPU，不公平）；比"零训练 + 无 GPU 依赖"。config F 单 tile ~20ms CPU；CNN 需 ~6h GPU 训练。

## 来源

- config F 计时：`repro/` 内联脚本（200 个真实 patch，Weihui）
- CNN 训练时长：`repro/train_unetpp.log` 逐 epoch 秒数求和
- CNN 推理：`repro/` GPU benchmark（7ms/forward）
