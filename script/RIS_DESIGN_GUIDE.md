# RIS Pattern 設計工具使用指引

> 這份文件整理 round 1-10 /loop 實驗後得出的「現在最有效的工作流程」，幫使用者
> 真實 use case 直接上手。

## 結論先行

對 **「為單一固定 target 找最佳 binary RIS pattern」** 的實際需求，最佳工具
是 **`design_pattern_for_target.py`**（with multi-restart + SA fine-tune）：

```bash
python script/design_pattern_for_target.py \
  --plateau_start 154 \
  --plateau_w 46 \
  --inc_theta 60 \
  --steps 1500 \
  --n_restarts 3 \
  --sa_steps 8000 \
  --sa_T0 20 \
  --sa_flip_n 3 \
  --device cuda:0
```

**2 分鐘**輸出可部署的 binary pattern + 評估報告。實測 suppression 可達
**+9.51 dB**（物理上限），即使 GD 全部 restart 卡 +2 dB local min，SA 也能
推到 +7+ dB（**round 15 驗證**）。比 generator-based 路線（v6 −0.46 dB）高
**+10 dB**。

## 工具樹

```
script/
├── design_pattern_for_target.py   ★ 為單一 target 找最佳 binary pattern
├── sweep_physical_limit.py        探物理可達區（多 target heatmap）
├── direct_pattern_search.py       驗證物理上限 + 對照 BinarySTE 訓練
├── post_quantize_eval.py          對 phase 1 連續 run 做後處理量化評估
├── train_multi_target.py          plan D：multi-target generator + cond_reg
├── train_direct_ris.py            plan E：無 surrogate、可微 RIS sim 訓練
├── pretrain_surrogate.py          surrogate 預訓練（含 --n_structured）
├── generate_structured_patterns.py 線性相位梯度結構化 pattern 工具
├── inspect_ris_run.py             trainer 訓練結果完整檢視
├── compare_ris_runs.py            多 run 跨 overlay 比較
├── run_bit_migration.sh           一鍵 phase1+phase2+inspect
└── run_full_v4.sh                 一鍵 pretrain+phase1+phase2+inspect
```

## 三種使用情境

### 情境 A：單一 target 部署（推薦）

```bash
# 1. 探物理可達區，挑容易的 target 配對（可選）
python script/sweep_physical_limit.py --device cuda:0

# 2. 為選定 target 設計 pattern（with multi-restart）
python script/design_pattern_for_target.py \
  --plateau_start 280 --plateau_w 46 \
  --n_restarts 5 --device cuda:0

# 3. 拿產出的 pattern_binary.npy 部署到硬體
```

### 情境 B：跨多 target 即時生成（research）

> ⚠️ Generator-based 路線經 11 個 run 驗證為**架構限制**，無法做到真正
> conditional。最佳結果 v6 仍是 hamming ~0.4%（10 target 給幾乎相同 pattern），
> 視為「對全部 target 通用最佳 pattern」而非「conditional 生成器」。

```bash
# 走 v6 配置（multi-target + cond_reg）
python script/train_multi_target.py \
  --epochs 500 \
  --cond_reg_weight 1.0 \
  --device cuda:0
```

### 情境 C：物理特性研究

```bash
# 大規模 sweep 畫 suppression heatmap
python script/sweep_physical_limit.py \
  --n_widths 6 --n_positions 12 \
  --steps 2000 --device cuda:0
```

## 物理上限（15×15 RIS, 28 GHz, 入射 θ=−40°/φ=90°）

從 round 9 sweep 32 runs：

| 指標 | 值 | 條件 |
|------|------|------|
| Suppression mean | +2.91 dB | 全 32 runs |
| Suppression max | +6.94 dB | plateau θ_center=+61.5° (寬 46) + 5 restarts |
| Suppression min | +0.31 dB | plateau θ_center=−55° (寬 20) |

**規律**：
- 寬 plateau（46+ samples）比窄（20）高 1-2 dB
- ±60° 兩端比 0° 容易（接近鏡面反射方向有優勢）
- 沒有 dead zones — 任何 target 都有 +0.3 dB 級別的解

## RIS 陣列大小建議（round 10 sweep）

| RIS 尺寸 | total cells | best suppression（3 target 最佳）|
|---------|-------------|--------------------------------|
| 10×10 | 100 | +4.27 dB |
| 15×15 | 225 | +4.61 dB（current default）|
| 20×20 | 400 | +5.70 dB |
| **25×25** | **625** | **+7.57 dB ★** |

**建議**：硬體允許下選 25×25 — 比 15×15 多 +3 dB suppression。15×15 在某些
target 反而比 10×10 還差（local-minima 物理特性，原因待查），不是最佳選擇。

## 入射角建議（round 12 sweep）

| inc_θ | left target | center | right | 平均 |
|-------|-------------|--------|-------|------|
| -60° | +6.84 | +5.88 | +6.20 | **+6.31** |
| -40° (default) | +5.32 | +3.24 | +7.04 | +5.20 |
| -20° | +4.07 | +5.33 | +4.34 | +4.58 |
| **+0°（垂直）** | +1.01 | +2.78 | +2.01 | **+1.93 最差** |
| +20° | +4.75 | +5.00 | +5.57 | +5.11 |
| +40° | **+8.45** | +4.75 | +4.53 | +5.91 |
| **+60°** | +5.00 | **+9.51** | +5.96 | **+6.82 最佳** |

**強烈建議**：硬體安裝時避免垂直入射（θ=0°），偏角 ±60° 平均高 +5 dB suppression。

## 頻率 × 陣列大小選型表（round 18 sweep）

| 部署頻率 | 推薦 size | mean (dB) | max (dB) | aperture |
|---------|-----------|-----------|----------|----------|
| 5.6 GHz | **19×19 ★ 或 13×13**（避開 15-17 valley）| +9.07 / +8.16 | **+11.82 ★** / +10.40 | 9.5λ / 6.5λ |
| 28 GHz | **13×13 ★**（valley 在 17×17）| +8.93 | +9.37 | 6.5λ=70mm |
| 60 GHz | **15×15 ★** | +8.55 | **+10.51** | 7.5λ=38mm |

**⚠️ Bimodal valley 隨 inc_θ 移動**（round 21-22 驗證）：

當 inc_θ=+60°，三頻率 valley 在 17×17（5.6 GHz +7.08, 28 GHz +7.43, 60 GHz +7.02）。
當 inc_θ=-40°，28 GHz valley 在 13×13 (+7.39)，17×17 反而是 peak (+7.97)。

**結論**：bimodal valley **不是純 size 物理特性**，是 **inc_θ × size 配對的
anti-resonance**。不同 inc_θ 下避開的 size 不同。

**強烈建議**：為實際部署 inc_θ 跑 sweep_frequency_x_size.py fine grid 找
你硬體配置下的真正最佳 size。下表只是 inc_θ=+60° 下的結果。

**規律**：
- 高頻（60 GHz）對 size 不敏感，10×10 已可達 +10.51 dB（歷史最高）
- 中頻（28 GHz）甜蜜點 7.5λ aperture
- 低頻（5.6 GHz）需要更大 element_num 補 aperture

歷史最高紀錄 +10.51 dB（60 GHz × 10×10, max across 3 seeds, GD+SA）。

## Generator-based 路線（11 個 run 完整對照）

| Version | Method | Suppression mean |
|---------|--------|------------------|
| v1 | 純 binary + pretrained surrogate | −4.08 dB |
| v2 | + 反 collapse 三 combo | −1.84 dB |
| v3 | + 位元遷移 | −2.21 dB |
| v4 phase1+postq | 結構化 surrogate + 後量化 | −1.63 dB |
| v4 phase2 | + binary fine-tune | −1.32 dB |
| v5 | plan D multi-target binary STE | −6.40 dB |
| **v6** | **plan D + cond_reg=1 (continuous)** | **−0.46 dB ★** |
| v7 | cond_reg=5（過強）| −2.69 dB |
| v8 | plan E direct RIS sim | −6.08 dB |
| v9 | plan E + plain MLP（無 noise）| −4.23 dB |
| **direct GD** | per-target，5 restart | **+6.94 dB** |

**關鍵診斷**（round 8）：v8 + v9 的 hamming = 0.00%（10 target 完全相同 pattern）
→ generator-based 對 conditioning failure 是**架構限制不是可修 bug**。
不論 surrogate 是否預訓練、是否用結構化資料、是否有 Gumbel noise，generator
為 N 個 target 妥協出單一 fixed pattern 是最終結果。

## 訓練 trainer 的 binary RIS 流程（如果要走完整 trainer）

```bash
# 1. 預訓練 surrogate（5000 random + 1000 structured）
python script/pretrain_surrogate.py --n_samples 5000 --n_structured 1000 --device cuda:0

# 2. 跑兩階段位元遷移
python -m antenna train +experiment=train_ris_phase1_continuous experiment_name=RIS-phase1
python -m antenna train +experiment=train_ris_phase2_binary experiment_name=RIS-phase2

# 3. inspect 出 binary 評估報告
python script/inspect_ris_run.py result/RIS-phase2

# 或一鍵跑：
bash script/run_full_v4.sh v4
```

但**這個流程無法達到 direct GD 的水準**，僅供研究 generator path 之用。
