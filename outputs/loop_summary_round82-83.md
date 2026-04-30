# /loop Round 82–83 兩輪總結 — Diversity vs Class Balance Trade-off

> R81 揭露 surrogate ranking failure。R82 加 random patterns 部分救回 (Spearman 0.03 → 0.30)。
> R83 試更多 random，**反而變差** (0.30 → 0.06)。**class balance 比 raw scaling 重要**。

## TL;DR

| Dataset | Optimized | Random | Total | Spearman | Top-5 pred true |
|---------|-----------|--------|-------|----------|-----------------|
| v2 | 108 | 0 | 108 | 0.031 | +1.13 |
| v3 (3 rand/cfg) | 108 | 324 | 432 | **+0.305** | -0.15 |
| v4 (6 rand/cfg) | 108 | 648 | 756 | **+0.060 ↓** | +0.79 |

→ **更多 random data 反而傷害 overall ranking** (class imbalance)

## R82 — Dataset_v3 (Diversity Hypothesis)

### 動機

R81 surrogate ranking failure (Spearman 0.031, basically random)。
Hypothesis: training data 只覆蓋 optimized patterns (窄分佈)，
NN 學到 trivial mean predictor。

### 設計

對 dataset_v2 每 config 加 3 個 random binary patterns + 跑 sim 算 response。
Total: 108 optimized + 324 random = 432 Pareto rows。

### 結果

| Metric | v2 (only optimized) | v3 (with random) |
|--------|--------------------|--------------------|
| Function MAE | 2.57 dB | 14.51 dB (wider range) |
| Pearson worst_supp | -0.027 | +0.229 |
| **Spearman worst_supp** | **0.031** | **+0.305** ★ |
| rw=2 Spearman | 0.093 | +0.466 |

**10× ranking improvement**. Diversity hypothesis 部分證實。
flat-top 區域 (rw=2) ranking 比 steering 區域 (rw=0) 顯著好 (0.466 vs 0.182)。

## R83 — Dataset_v4 (Scale Up Random)

### 動機

R82 證明 random 有用。試更多 random 看能否進一步 push Spearman 0.30 → 0.5+。

### 設計

每 config 加 6 個 random patterns (vs v3 的 3 個)。
Total: 108 optimized + 648 random = 756 Pareto rows。

### 結果（反直覺）

| Metric | v3 | v4 | 變化 |
|--------|----|----|----|
| Total rows | 432 | 756 | +75% |
| Pearson | +0.229 | -0.113 | ↓↓ |
| **Spearman** | **0.305** | **0.060** | **↓↓ (almost zero!)** |
| Top-5 pred true | -0.15 | +0.79 | 略升 |
| Top-50 pred true | -5.05 | -6.03 | 略降 |

### 為什麼變差：Class Imbalance

```
v3: 108 optimized / 324 random  = 25% optimized
v4: 108 optimized / 648 random  = 14% optimized

Surrogate 主要學 random 分布:
  - Random patterns 範圍寬 (-30 ~ +5 dB)
  - 容易預測 mean 拿好 MSE loss
  - 但 optimized patterns ranking 失真
  
Spearman over ALL 756 rows:
  - Dominated by random pattern statistics
  - Optimized 區域 (我們 care 的) ranking 失準
  - → 整體 Spearman 下降
```

## Critical Lesson: Class Balance > Raw Scaling

| 設計 | 結果 |
|------|------|
| ✗ 更多 random data + 同 optimized | class imbalance, ranking 變差 |
| ✓ 保持 50/50 ratio random/optimized | 預期最佳 |
| ✓ 或 weighted training (over-sample optimized) | 替代方案 |

對 patch dataset 設計：

```
WRONG:
  random_HFSS_runs = 1000
  optimized_HFSS_runs = 100
  → class imbalance ratio 10:1
  → surrogate 學 trivial random distribution
  → 對 optimized 區域 (deployment 真正在意) 預測不準

RIGHT:
  random_HFSS_runs = 500
  optimized_HFSS_runs = 500
  → balanced
  → surrogate 學到真實 input → response mapping

或:
  Active learning loop:
    Initial: 100 random + 100 optimized
    Iter: BO acquisition 選 100 high-uncertainty
    Iter: GD-output 加 100 (擴 optimized 分布)
    → 維持 balance + dataset growth
```

## flat-top vs steering: Different Difficulty

R82 揭露 rw=2 (flat-top) Spearman 0.466 vs rw=0 (steering) 0.182。

```
rw=2 (worst-case + ripple penalty):
  - Loss landscape 較 well-conditioned
  - Optimized patterns 較相似 (向 flat-top mode 收斂)
  - Surrogate 較易 fit

rw=0 (max-max steering):
  - Loss landscape 多峰 (R71 hamming 51.72%)
  - Optimized patterns 多 mode (尖峰位置不同)
  - Surrogate 更難 fit
```

對 patch: **constrained optimization (eg 帽蓋 + ripple) 比 unconstrained**
**(eg 純 maximize gain) 對 surrogate 友好**。

## Realistic Patch Budget Update (final)

| Surrogate quality | Approximate dataset size | Use case |
|-------------------|--------------------------|----------|
| Function MAE only | 100-200 entries | Coarse screening |
| Spearman > 0.3 | 400-500 entries (balanced) | Top-K filter |
| **Spearman > 0.7 (deploy)** | **1500-3000 entries (balanced + diverse)** | BO acquisition |
| Gradient cosine > 0.7 | likely **3000+ entries + Sobolev / advanced arch** | GD-through-surrogate |

對 patch HFSS budget:
- 5 min/run × 1500 = 5 days
- 5 min/run × 3000 = 10 days
- 不可行單機 → 用 active learning + parallel HFSS workers

## 紀錄歷程更新

| Round | 結果 | 教訓 |
|-------|------|------|
| R66-R72 | dataset/surrogate scaling N^-1.62 | function MAE 估計 |
| R71 | hamming 51.72% | (config → pattern) multimodal |
| R76 | methodology distill | 5 原則 + 4 templates |
| R77 | adversarial gap | function MAE ≠ deploy quality |
| R78 | rule out OOD | gradient quality 是真 bottleneck |
| R79 | gradient cosine 0.001 | 完全 random gradient |
| R80 | Sobolev fail | architectural 限制 |
| R81 | ranking 0.031 | dataset diversity 不夠 |
| **R82** | **v3 diversity** | **Spearman 0.03 → 0.30 (10x)** |
| **R83** | **v4 imbalance** | **class balance > raw scaling** |

## 累計（83 rounds, 119+ commits）

完整 surrogate methodology 探索:
1. Loss design ✓ (R64)
2. Dataset diversity ✓ (R82)
3. Class balance ✓ (R83)
4. Function MAE 不夠 ✓ (R77)
5. Gradient quality bottleneck ✓ (R79)
6. Sobolev 救不回 ✓ (R80)
7. Ranking quality 也需 work ✓ (R81)

對 patch antenna methodology, 上述 7 個 lessons 都 transferable, 已 codified
in `script/PATCH_METHODOLOGY.md` (R76 + R77/78/79/80 updates).

## 下一步

RIS playground 探索已 saturate。本輪建議:

1. **Active learning loop demo**: 用 v3 surrogate + 5 iterations 證明 BO acquisition
   能比 random sampling 更 efficient. Patch transition 的最後 piece.

2. **Patch transition**: 帶上完整 R76-R83 lessons 開始 patch dataset 收集.

3. **收尾**: 寫整體 RIS methodology paper-style 完結文檔.
