# M1：split-based（分块 Otsu）基线 vs 固定闸门

> 目的：回答审稿人"F 只是阈值+连通域，比 30 年前的业务链好在哪里"。
> 协议与 `_eval_3event.py` 完全一致：testing_case_256、保留 (GT>0)>=100 的 patch、
> 像素聚合、Jubba 合并（混淆矩阵相加）、open 口径（GT==1 为正类）。
> 复现：`python _eval_split_baseline.py`、`python _eval_split_baseline_b.py`

## 结果 A：分块 Otsu 暗水（不含变化策略）

| 配置 | Weihui | Nova | Jubba | **均值** |
|:--|--:|--:|--:|--:|
| A（逐景 Otsu，无闸门） | 0.096 | 0.069 | 0.489 | **0.218** |
| F（逐景交集 + 闸门） | 0.773 | 0.367 | 0.822 | **0.654** |
| S16（块 16，无闸门） | 0.079 | 0.062 | 0.323 | 0.155 |
| S32（块 32，无闸门） | 0.084 | 0.066 | 0.353 | 0.167 |
| S64（块 64，无闸门） | 0.089 | 0.068 | 0.386 | 0.181 |
| S128（块 128，无闸门） | 0.093 | 0.069 | 0.422 | 0.195 |
| S16 + 闸门 | 0.649 | 0.133 | 0.639 | 0.474 |
| S32 + 闸门 | 0.661 | 0.134 | 0.664 | 0.486 |
| S64 + 闸门 | 0.672 | 0.135 | 0.683 | 0.497 |
| S128 + 闸门 | 0.679 | 0.135 | 0.694 | 0.503 |

**结论**：
1. 分块阈值本身**弱于**逐景阈值（0.155–0.195 vs 0.218）——与文献中 split-based 的已知弱点一致。
2. 加同样的两个常数后，**每个块尺寸都提升约 3 倍**（0.155→0.474、0.195→0.503）→ 闸门的作用与阈值粒度无关。
3. S+F 仍低于 F（0.47–0.50 vs 0.654），因为该基线**不含变化策略**（S_int ∩ S_chg），差的这部分来自第二路证据，不是阈值粒度。

## 结果 B：分块交集（含变化策略）

| 配置 | Weihui | Nova | Jubba | 均值 |
|:--|--:|--:|--:|--:|
| B32 | 0.049 | 0.047 | 0.219 | 0.105 |
| B64 | 0.048 | 0.046 | 0.224 | 0.106 |
| B32 + 闸门 | 0.274 | 0.092 | 0.325 | 0.230 |
| B64 + 闸门 | 0.243 | 0.089 | 0.298 | 0.210 |

**说明**：分块做法在差分图上极不稳定（块内 Otsu 在 |ΔI| 上经常切错），因此"分块交集"不是经典业务链的做法，
**不作为论文中的基线使用**；结果 A 才是与 Martinis 式 split-based 对应的构造。

## 论文中的写法（建议）

> Split-based thresholding — the classic operational configuration — does not close the gap:
> with block-wise Otsu in place of per-scene thresholds the ungated mask scores 0.155–0.195 over
> block sizes 16–128 px, and the same two constants raise every setting to 0.474–0.503, a
> three-fold gain that reproduces the gate effect independently of how the threshold is chosen.
