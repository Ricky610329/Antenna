# /loop Round 68–69 兩輪總結 — Surrogate Proof of Concept + Critical Negative Findings

> R66-R67 建 dataset_v1 (72 entries)。R68-R69 試訓 surrogate，揭露兩個對 patch
> 移植非常重要的負面結果。

## TL;DR

| 實驗 | 結果 | Lesson |
|------|------|--------|
| R68 MLP forward (361-dim) | 嚴重 overfit (train_mse 1.2, test 75) | spatial inductive bias 必要 |
| **R68 CNN forward (361-dim)** | **MAE 4.89 dB worst_supp** | CNN 砍誤差一半，但仍不足 |
| **R69 CNN metric (6-dim scalar)** | **collapsed to mean predictor** | sparse supervision 不夠 |

→ **Surrogate 應預測 dense output（full S-curve），不是 scalar metrics**

## R68 — Forward Surrogate (預測 361-dim response)

### Architecture comparison

```
MLP: pattern_flat (1681) + mask_flat (1681) + config (6) → response (361)
     2.5M params, train→1.2 (overfit), test=75, MAE 11.64 dB

CNN: pattern_2d + mask_2d + config_broadcast → conv → response (361)
     ~150K params, train=52, test=59, MAE 4.89 dB
```

CNN 用 spatial inductive bias 把誤差砍半。但 4.89 dB 仍離 patch deployment
standard (< 1 dB) 遠。**72 entries 對 forward surrogate 不足。**

### 系統性 bias

CNN 在 rw=0 (high-suppression) cases 系統性 under-predict（學到太多 rw=2 模式）：
- idx 0: true +1.85, pred -5.76 (under by 7.6)
- idx 4: true +3.65, pred -7.83 (under by 11.5)
- idx 12: true +1.95, pred -4.81 (under by 6.8)

→ Dataset 需要 class balance + 更多多樣性。

## R69 — Metric Surrogate (改預測 scalar metrics)

### 動機
Forward surrogate 預測 361-dim response 對 72 examples 太 sparse。
試只預測 6 個 scalar (worst_supp, ripple, ...) 是否更易學。

### 結果：完全失敗

```
metric         | true_std | MAE  | MAE/std
worst_supp     | 1.72     | 1.52 | 0.89
ripple         | 6.16     | 6.13 | 0.99
side_max       | 7.65     | 7.58 | 0.99
main_min       | 6.16     | 6.13 | 0.99
```

MAE/std ≈ 1.0 → 模型預測訓練均值。所有 test 預測 worst=+0.25, ripple=8.80
(均值)。完全沒學到 (pattern × config) → metrics 的 mapping。

### 為什麼？

| Surrogate | output dim | supervision per example | result |
|-----------|-----------|------------------------|--------|
| Forward (R68 CNN) | 361 | dense | **MAE 4.89** ✓ |
| Metric (R69 CNN) | 6 | sparse | **mean collapse** ✗ |

Sparse supervision 在小 dataset 下不夠 regularize，CNN 容易 trivially
解 "predict mean"（mean MSE 已經很小）。Dense supervision 的每維 prediction
都是有意義的 signal，自然 regularize.

## 對 Patch Antenna Surrogate 的 Critical Methodology

### Lesson 1: Predict full curve, not scalars

```
GOOD:  geometry × freq → S11(f) full curve [hundreds of points]
                         ↓ post-process
                         worst-case S11, BW, gain peaks

BAD:   geometry × freq → (worst_S11, BW, gain) scalars [collapsed predictor]
```

### Lesson 2: CNN architecture for 2D geometry

- MLP flatten 掉空間結構 → overfit
- CNN preserves locality → 必要

對 patch 直接適用：patch geometry 也是 2D（patch shape, slot position, etc.）。

### Lesson 3: Class balance & diversity

R68 surrogate systematic bias on rw=0 揭露 dataset imbalance（rw=0 vs rw=2
雖數量平衡，但 difficulty 不對等）。

對 patch: dataset 需 explicit cover edge cases (extreme S11, narrow BW, etc.)。

## 紀錄歷程修正

| 階段 | 焦點 | 紀錄/結果 |
|------|------|------|
| R57-R63 sigmoid path | max-max steering | +30.99 (虛胖) |
| R64-R65 worst-case loss | Pareto frontier | +6.88 honest worst |
| R66-R67 dataset_v1 | 72 entries Pareto | 67% flat-top at rw=2 |
| **R68 CNN forward surrogate** | **proof of concept** | MAE 4.89 dB |
| **R69 metric surrogate** | **negative result** | mean collapse |

## R69 進度

dataset_v2 (54 configs × 4 = 216 runs，含 n=41) 在背景跑。10/54 完成。
預計 60+ min 跑完。R70+ 會用 v2 重訓 surrogate 看 dataset scaling 影響。

## Open Questions（更新）

1. dataset_v2 (~5x v1) surrogate MAE 能否從 4.89 → < 2 dB？
2. Active learning 比 random scaling 多 sample-efficient 多少？
3. 該訓 (config → pattern) 反向 generator 還是 (pattern → response) 正向 surrogate？
4. Forward surrogate + GD-through-surrogate 能否取代 sigmoid path？
5. **Patch transition 的 dataset size 經驗值**: R68-R69 暗示 patch 也需 200-500 entries 起跳

## Sources

- R68 forward CNN: [paper "Continuous Functional Learning"]
- R69 negative result: [paper "Sample Complexity of Implicit Neural Representations"]
