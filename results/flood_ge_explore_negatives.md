# 验证门改进 · 探索负结果（归档，不进论文正文）

> 记录：2026-09-06 | 脚本：`repro/_diag_gate_coh.py`
> 问题：能否在"独立证据验证门"框架内、不破坏主口径的前提下，把 config F 的门本身做得更强？
> 结论：**开放口径的门已接近合理工作点；相干度是正交的城区证据，加进开放门只添 FP。真改进（城区分支）是换目标函数，非门提升。**

---

## 假设
把相干损失（Δγ = pre_coh − co_coh）作为第三条独立证据引入开放验证门（G-A：作 −12dB 替证；G-C：三证据 ≥2-of-3），或开一条相干城区分支（G-B）——依据文献"相干度补强度失效场景"。

## 协议
与主口径一致：testing_case_256、≥100 洪水像元过滤、config F 掩膜（int∩chg ∩ postVV<−12dB ∩ conn≥500）、Jubba 合并、open 像素聚合 F1。对 TP/FN/FP（开放，GT==1 正类）与城区 GT（GT==2）像素统计 Δγ 分布。

## 结果（Δγ 中位数 / >0.2 占比）

| 事件 | TP | FN | FP | 城区 GT |
|---|---|---|---|---|
| Weihui | 0.09 / 27% | 0.09 / 27% | 0.07 / 22% | **0.36 / 82%** |
| NovaKakhovka | 0.01 / 20% | 0.01 / 23% | −0.02 / 7% | **0.40 / 92%** |
| Jubba_1 | 0.03 / 8% | 0.03 / 8% | 0.02 / 9% | **0.54 / 99%** |
| Jubba_2 | 0.01 / 0% | −0.01 / 0% | −0.01 / 1% | **0.43 / 93%** |

## 否决理由
1. **相干度是"城区"证据，不是"开放洪水"证据**：城区 GT 的 Δγ>0.2 达 82–99%；开放洪水的 TP/FN 在所有事件上 Δγ 低（Jubba ≤8%）。物理上一致——城区双反弹亮信号与暗水门互斥，相干是独立于强度-暗水的机制。
2. **G-A 无 recall 收益且有 FP 代价**：Weihui FN 与 TP 的 Δγ 相同（0.09 vs 0.09）——相干救不回被 −12dB 误杀的召回；Weihui FP 22%>0.2，Nova 1.7M FP 中 7%>0.2 → "OR S_coh" 净添假阳。
3. **G-C 同理**：Jubba 开放水本身低相干（TP 仅 8%>0.2），2-of-3 只加噪。
4. **Jubba_2 / Nova 是精度问题不是召回问题**（P=0.064 / 0.262，FP≫TP），且 FP 的 Δγ 偏低——相干 AND 只误伤 TP。

## "城区分支 = 换目标函数"
- 城区 GT 极小：Nova 194 – Weihui 6,184 像素（Weihui 1.7%、Nova 0.02% 的洪水）。
- 主口径 open = GT==1 正类、任何洪水预测为正例（class-vs-rest）→ **城区预测天然计入 open 的 FP、压低 0.654**。
- 因此"config F 报城区"= 双类输出 + 主口径/城区表/叙事整套重写，**不是加一行**。
- 留作投稿后 / 期刊扩展版；若未来报双类 config，需另起主口径。

## future-work 一句话（可进论文讨论节，溯源本文件）
> "Coherence loss is orthogonal to the intensity-dark-water signal and acts as a specific urban-flood marker in this SLC benchmark; fusing it into the open-water gate only adds false positives, so an urban-capable configuration would require a separate verification branch and a revised evaluation semantics (deferred)."

## 未开战线（backlog，非本期）
- G-B（相干城区分支）：需接受主口径重定。
- Nova / Jubba_2 假阳诊断（精度方向）：ROI 低——可动 margin 小（Nova +0.058 vs CNN；Jubba_2 已定边界），新旋钮要重走校准/敏感性/表重出；记 backlog。
