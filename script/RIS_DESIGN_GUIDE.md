# RIS Pattern 設計工具使用指引

> 這份文件整理 round 1-10 /loop 實驗後得出的「現在最有效的工作流程」，幫使用者
> 真實 use case 直接上手。

## 結論先行

對 **「為單一固定 target 找最佳 binary RIS pattern」** 的實際需求，最佳工具
是 **`design_pattern_for_target.py`**（with multi-restart）：

```bash
python script/design_pattern_for_target.py \
  --plateau_start 280 \
  --plateau_w 46 \
  --steps 1500 \
  --n_restarts 5 \
  --device cuda:0
```

**5 分鐘**輸出可部署的 binary pattern + 評估報告。實測 suppression 可達
**+6.94 dB**（接近物理上限），比 generator-based 路線（v6 −0.46 dB）高
**+7.4 dB**。

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
