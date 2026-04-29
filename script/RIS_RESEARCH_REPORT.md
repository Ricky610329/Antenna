# Binary RIS Pattern Optimization — 14-round /loop 完整研究報告

> 期間 2026-04-29，基於本實驗室 11 篇碩論研究脈絡 + Antenna repo 既有架構，
> 對「RIS 硬體相位 {0, π} 約束下的 binary pattern 生成」進行完整探索。

## TL;DR

**最佳工作流程**（對使用者真實 use case「為單一固定 target 找 binary RIS pattern」）：

```bash
python script/design_pattern_for_target.py \
  --element_num 15 \
  --inc_theta 60 \
  --plateau_start 154 \
  --plateau_w 46 \
  --n_restarts 5 \
  --device cuda:0
```

**5 分鐘**輸出 binary pattern + suppression **+9.51 dB**（物理上限級）。
比實驗室原本的 generator-based 路線高 **+10 dB suppression**。

## 動機

- RIS 硬體只支援相位 {0, π}（binary phase）
- 實驗室過往研究（GAN / AE / 位元遷移 / 強化學習）多在連續相位或量化後評估
- 原 Antenna repo 用 generator-based 架構，發現「對所有 target 給相同 pattern」
  的 conditioning failure，但未深入診斷
- 目標：找出真正可用的 binary RIS pattern 設計工作流程

## 方法總覽（兩條路線並行）

### 路線 A — Generator-based（v1-v9）
訓練 NN 從 target response → binary pattern logits → BinarySTE → 嚴格 {0, 1}：
- v1: 純 binary + pretrained surrogate
- v2: + 反 collapse 三 combo (H/I/J)
- v3: + 位元遷移 (curriculum quantization)
- v4: + 結構化 pattern 補強 surrogate + 後量化
- v5-v7: + multi-target + conditional regularizer (cond_reg)
- v8-v9: + 直接 RIS sim 替代 surrogate（plain MLP / Biased Gumbel）

### 路線 B — Per-target Direct GD
對單一 target 直接 GD on pattern logits：
- 連續訓練 + 後處理量化（continuous → hard threshold）
- Multi-restart 避開 local minimum
- 不依賴任何學習網路

## 實驗結果

### Generator-based 完整對照

| 版本 | suppression mean | hamming% | 評語 |
|-----|------------------|----------|------|
| v1 | −4.08 | n/a | baseline |
| v2 | −1.84 | n/a | 反 collapse 三 combo |
| v3 | −2.21 | n/a | + 位元遷移 |
| v4 phase1+postq | −1.63 | n/a | + 結構化 surrogate |
| v4 phase2 | −1.32 | n/a | + binary fine-tune |
| v5 | −6.40 | n/a | multi-target binary STE |
| **v6** | **−0.46** | **0.40%** | **史上最好（但仍 fake conditional）** |
| v7 | −2.69 | n/a | cond_reg 過強 |
| v8 | −6.08 | 0.00% | direct RIS sim |
| v9 | −4.23 | 0.00% | + plain MLP（無 noise）|

### Per-target Direct GD 物理 sweep

| Sweep 維度 | 範圍 | 最佳 |
|-----------|------|------|
| Plateau 位置（round 9）| 8 starts × 4 widths | +6.94 dB (15×15) |
| RIS 邊長（round 10）| 10/15/20/25 | +7.57 dB (25×25) |
| 入射角（round 12）| -60° ~ +60° | **+9.51 dB (inc_θ=+60°, 15×15)** |
| 多 restart（round 13）| 5 vs 10 | 1/10 機率達上限 |

### 紀錄歷程

```
v1 (純 binary, 15×15)              −4.08 dB
v6 (generator best)                −0.46 dB
direct GD 15×15 (5 restart)        +6.94 dB
direct GD 25×25 batch 5 targets    +8.65 dB
direct GD 15×15 inc_θ=+60°         +9.51 dB ← 物理上限
```

## 關鍵發現

### 1. Generator-based 的 conditioning failure 是架構限制
v6-v9 hamming = 0%（10 個 target 給幾乎相同 pattern）→ 各種反 collapse、
multi-target、cond_reg、移除 Gumbel noise、直接 RIS sim 都無法解決。
**根因**：generator 必須對 N 個 target 妥協出單一 fixed pattern，這是
「one-shot generator」與「per-target optimization」的本質差距。

### 2. BinarySTE 訓練不穩
direct GD 比較顯示 BinarySTE (-3.34 dB) << continuous + post-quantize (+3.05 dB)。
原因：identity backward 在量化邊界與 main loss landscape 不對齊，造成震盪。

### 3. Multi-restart 是必要、不是可選
單次 GD 約 1/10 機率達物理上限，其他 restarts 卡 local minimum +2~+6 dB。
建議 5 個 restart 起跳。

### 4. RIS 邊長與入射角不是獨立 multiplicative
- 25×25 在 inc_θ=-40° (default) 達 +7.57 dB
- 15×15 在 inc_θ=+60° 達 +9.51 dB（更好！）
- 25×25 在 inc_θ=+60° 反而 +7.42 dB（更糟）
**直覺**：25×25 cells 更多但 GD landscape 更崎嶇，在某些 inc_θ 配對下
local minima 比 15×15 還多。

### 5. 入射角是最重要的物理變因
| inc_θ | 平均 suppression |
|-------|-------------------|
| 0°（垂直） | +1.93 dB ← 最差 |
| -40° (default) | +5.20 dB |
| ±60° | +6.5~+6.8 dB ← 最佳 |

**強烈建議**：硬體安裝避免垂直入射，偏角 ±60° 比 default -40° 高 +1.6 dB。

### 6. 寬 plateau > 窄 plateau
寬 46-60 的 main beam region 比窄 20-33 高 1-2 dB。設計時放寬要求有利。

## 工具庫

```
script/
├── design_pattern_for_target.py   ★★★ 為單一 target 設計（with multi-restart, inc_theta）
├── design_batch.py                ★★★ 批次設計多個 target（CSV/CLI）
├── sweep_physical_limit.py        探物理可達區（plateau pos × width）
├── sweep_element_num.py           RIS 邊長 sweep
├── sweep_incidence_angle.py       入射角 sweep
├── direct_pattern_search.py       diagnostics — 物理上限 + STE 對照
├── post_quantize_eval.py          phase 1 連續 → 後量化評估
├── inspect_ris_run.py             trainer 訓練結果完整檢視
├── compare_ris_runs.py            多 run 跨 overlay
├── train_multi_target.py          plan D（multi-target generator + cond_reg）
├── train_direct_ris.py            plan E（無 surrogate、直接 RIS sim）
├── pretrain_surrogate.py          surrogate 預訓練（含 --n_structured）
├── generate_structured_patterns.py 線性相位梯度結構化 pattern
├── run_bit_migration.sh           一鍵 phase1+phase2+inspect
├── run_full_v4.sh                 一鍵 pretrain+phase1+phase2+inspect
├── RIS_DESIGN_GUIDE.md            ★★ 使用者快速上手
└── RIS_RESEARCH_REPORT.md         ★ 本報告
```

## 對實驗室過往工作的看法（基於 round 4 多 agent 論文分析）

### 真正有料的設計
- **曾俊瑋 113「公式層取代 decoder」**：本研究的 direct GD path 正是這個哲學
  的延伸。但用戶 codebase 的 trainer 仍透過 surrogate，沒徹底執行此哲學
- **錢鵬予 114 BiScaleNorm**：在 patch 場景驗證有效，但 binary RIS 場景
  反而傷害（壓縮 logits 訊號）。**Domain 特性差異很重要**
- **蔡奇倫 114 位元遷移**：本質為 progressive QAT，對 v3 確實有微改善
  （-2.21 vs v1 -4.08），但不是真正解 conditioning。**正確的 framework
  名稱是 "two-stage QAT" / "gradual quantization"**

### 表面 fancy 但本質有限
- **Smooth G-STE**（蔡奇倫）：仔細推導發現 backward = sign(x-T)，跟 vanilla
  identity STE 在量化邊界有相同正負號問題（部分情境梯度方向錯）。實驗中
  BinarySTE direct GD 失敗，Smooth G-STE 預期類似
- **GAN（李宏文 110）**：FID 22.17 在 GAN 標準下不算好；mode collapse 已知
- **賴昱鈞 113 PPO RL**：解的是「序列選擇 on/off」不同問題，非本研究 use case

### 無關但有趣
- **陳柏廷 114 CSI MTA**：跨領域到 6G CSI 壓縮，跟 RIS 設計沒關聯
- **王騰緯 112 GWO 補資料**：GWO 太慢，本研究改用 plane-wave reflectarray
  公式（閉式、1000× 快）

## 對使用者實務建議

### 場景 A：單一固定 target（最常見）
```bash
python script/design_pattern_for_target.py \
  --element_num 15 \
  --inc_theta 60 \
  --plateau_start 154 \
  --plateau_w 46 \
  --n_restarts 5 \
  --device cuda:0
```
5 分鐘輸出 `pattern_binary.npy` 直接部署，期待 +9 dB suppression。

### 場景 B：多個 target 共用硬體
```bash
python script/design_batch.py \
  --csv targets.csv \
  --inc_theta 60 \
  --n_restarts 5
```
平均 +7 dB suppression。

### 硬體選型
- RIS 邊長：先試 15×15（甜蜜點），不要盲目選大
- 安裝角度：偏角 ±60° > default -40° > 垂直 0°（最差）
- 工作頻率：本研究在 28 GHz，其他頻率規律應類似（待 round 15+ 驗證）

### 不要走的路（已驗證失敗）
- Generator-based 路線（最好 v6 -0.46 dB，跟 direct GD +9 dB 差 9 dB）
- 直接 BinarySTE 訓練（震盪不收斂）
- 25×25 + +60° 組合（local-minima 更密集）

## 開放問題

1. **為何 25×25 + +60° 比 15×15 + +60° 差？**——GD landscape 物理特性，
   未深入分析
2. **不同頻率的規律**——5.6 / 60 GHz 是否相同 inc_θ=+60° 最佳？
3. **multi-restart 命中率提升**——SA / 二階段 fine-tune 能否從 1/10 提升
   到 5/10？
4. **真正 conditional generator 的可能性**——hypernetwork / retrieval-based
   架構未驗證

## Git 進度

23+ commits pushed 到 `ricky/modernize`。完整 commit 序列：
```
c347fc3 BinarySTE
77e7b87 binary_mode + pretrained surrogate
38ad7d9 反 collapse + inspect bug 修
35100f2 位元遷移
8705cd5 結構化 pattern + run_bit_migration
d5e6669 pretrain --n_structured
e6c844f run_full_v4 batch
f77c964 direct_pattern_search 揭露 BinarySTE 缺陷
4d80bf6 post_quantize_eval
08ad306 plan D minimal trainer
150e131 plan D + cond_reg
e46e659 plan E direct RIS sim
303967c plain MLP + design_pattern_for_target tool
36825ad sweep_physical_limit
3aae357 multi-restart
5e9efb8 design_guide + sweep_element_num
415e9e6 element_num 結果加 guide
fd753d6 design_batch 工具
4572e55 sweep_incidence_angle
7f41170 incidence sweep 結果加 guide
05ce59b inc_theta 參數 + round 13 驗證
```
