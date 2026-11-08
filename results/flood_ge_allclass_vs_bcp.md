# config F vs BCP：口径稳健性验证（open vs all-class）

> 日期：2026-08-21 | 目的：消除"BCP 对比口径不齐"质疑
> BCP per-site 数值取原论文最优后处理（t=0.2, w=9）下的每站点 F1（全类别口径、一年期 GRD 时序）；协议不同（per-site/GRD vs 我们 patch/SLC），非严格头对头，胜负仅供方向性参考。

## config F（post_vv<−12dB, conn≥500）两口径像素聚合 F1

| 场景 | open F1 (GT==1) | all F1 (GT>0) | BCP per-site | all 口径胜负 |
|:--|:--:|:--:|:--:|:--:|
| Weihui | 0.773 | 0.766 | 0.76 | config F 赢 |
| NovaKakhovka | 0.367 | 0.367 | 0.41 | config F 输 |
| Jubba（合并）| 0.822 | 0.821 | 0.65 | config F 赢 |
| **3 事件均值** | 0.654 | 0.651 | — | Weihui/Jubba 胜 |

## 结论

1. **all 与 open 几乎一致**（差值 ≤0.007）：测试场景城区类像素占比极小，正类定义对混淆矩阵几乎无影响。
2. **vs BCP 的胜负方向与口径无关**：两口径下均为 Weihui/Jubba 高于 BCP、Nova 低于 BCP（非严格头对头，仅方向性参考）。
3. **BCP 对比不再是口径漏洞**：无论审稿人要求 open 还是 all，结论相同。

> 口径稳健性证据可直接用于论文："The comparison against BCP is robust to the class protocol: all-class pixel-aggregated F1 values (Weihui 0.766, NovaKakhovka 0.367, Jubba 0.821) differ from the open-class values by less than 0.01."
