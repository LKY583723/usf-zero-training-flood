# UrbanSARFloods CNN Baseline 复现 · 最终结果

> 完成日期：2026-08-15
> 目标：对齐论文（Zhao 2024, CVPRW）的 CNN baseline 测试 F1（开放 F1 0.51–0.77，为论文在 Weihui/Jubba 上的 FO 范围；Nova 明显更低）

> ⚠️ **2026-08-27 纠正**：本文件第 10/37 行"z-score 复现仅 0.32–0.36"系**早期评测协议 bug 的误判**（评测脚本未做归一化，把原始 dB 值直接喂模型）。正确结论：z-score 训练（batch6）评测 open F1 = **0.553**，min-max 训练（batch12）= **0.518**；`data_norm.txt` 的两种归一化都是合法参数，**原论文未指定用哪种**。统一 zscore 基线方案见 `flood_ge_draft.md` §III-B 与 `ckpts_zscore/`。
>
> **最终基线（2026-08-29，统一 z-score + seed42，见 `flood_ge_final.md`）**：Unet++ **0.555**（最强）、Unet 0.528、DeepLabV3+ 0.462；unet++ 3-seed 均值 0.518±0.034。本文件表格为历史记录（2026-08-15，min-max/batch12 单 unet）。

## 复现配置

- **模型**：SMP Unet(resnet34, in_channels=8, classes=3)，从头训练（SMP 默认初始化；原论文声称 Xavier 但其公开仓库确认无自定义代码，见 `flood_ge_draft.md` §III-B）
- **训练**：100 epochs, batch=12, Adam lr=1e-5 wd=1e-4, WCE loss
- **归一化**：min-max（data_norm.txt 的 min_in/max_in）
- **评测**：testing_case_256 独立 256² patch + **像素聚合 F1**（class-level 混淆矩阵，对齐论文）
- **数据**：train 4501 + valid 1970（论文划分）；测试 = 论文测试事件

## 测试集结果

> ⚠️ 协议说明（2026-08-19 核对）：本表 Weihui/NovaKakhovka 为**整景（full-scene）评测**（见 `flood_repro_all4.md`/`flood_repro_pixelagg.md`），Jubba 因整景 6GB 下载受阻采用 patch 评测（0.756）。**论文对比统一采用 256² patch 口径**（`flood_ge_final.md`），见下方"统一口径"表。

**历史记录（整景/混合协议，供追溯）：**

| 测试事件 | FO（开放）F1 | FU（城区）F1 | 协议 |
|:--|:--|:--|:--|
| **Jubba_1** | **0.756** | 0.003 | patch（整景 403） |
| **Weihui** | 0.490 | 0.037 | 整景 |
| **NovaKakhovka** | 0.351 | 0.008 | 整景 |

**统一口径（256² patch · 像素聚合 open F1，与规则管线同协议，论文采用）：**

| 测试事件 | FO（开放）F1 | 城区 F1 | 论文参考 |
|:--|:--|:--|:--|
| **Jubba_1** | **0.756** | 0.003 | FO 0.51-0.77 ✓ |
| **Weihui** | **0.542** | 0.012 | FO 0.51-0.77 ✓ |
| **NovaKakhovka** | **0.258** | 0.000 | FO 低 ✓ |
| **Jubba_2** | **0.490** | 0.050 | — |

## 关键经验（复现踩坑记录）

1. **归一化**：本项目用 min-max（⚠️ 原"z-score 仅 0.32-0.36"结论已于 2026-08-27 纠正，见文件头注记）
2. **评测口径**：像素聚合 vs patch 平均差异巨大（Weihui 0.49 vs 0.32）；论文用 class-level 聚合
3. **patch = 整景 256 网格**：testing_case_256 与整景窗口数值一致（diff=0），可从整景切
4. **训练 batch=12**：对齐论文 BN 统计（batch 6 的 BN 噪声大）
5. Jubba 整景 6GB 下载受阻（hf CDN 403）→ 用 1000 个 2MB patch + 整景 GT 切坐标

## 对比 config F（零训练）

| 方法 | Jubba_1 FO F1 | Jubba_2 FO F1 | Weihui FO F1 | 类型 |
|:--|:--|:--|:--|:--|
| 复现 CNN（Unet） | **0.756** | 0.490 | 0.490 | 有训练 |
| 论文 CNN | 0.51–0.77 | — | 0.51–0.77 | 有训练 |

> 论文 CNN 的 0.51–0.77 为其 FO F1 在 Weihui 与 Jubba 上的范围（Nova 明显更低，precision<0.2，不在该区间）。
| **config A** | **0.538** | — | 0.067 | 零训练 |
| config C（Qwen 路由）| 0.531 | — | 0.068 | 零训练 |
| config F | 0.536 | — | 0.091 | 零训练 |

## 关键结论

1. **零训练 config F 在 Jubba（纯开放洪水）FO F1 0.538**，接近复现 CNN 0.756——开放洪水的暗水阈值策略直接命中
2. **config C/F 在 Jubba 上无提升**（略降）：开放洪水不需要路由，int 已最优
3. **路由的价值在混合场景**（Weihui 有城市+开放），对纯开放无增益
4. **规则管线 vs CNN 差距集中在城区**：Jubba 0.54 vs 0.76（小差距），Weihui 0.09 vs 0.49（大差距）——因为 Weihui 城区洪水占比高
5. **城区洪水是所有方法的短板**（CNN 0.03-0.05，config F 0.00）——相干度策略改进是规则管线下一步突破的关键
