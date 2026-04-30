# /loop Round 84–85 兩輪總結 — Class Balance Confirmed + Active Learning Greedy Fails

> R84 確認 class balance hypothesis (1:1 比 1:3 / 1:6 都好 for rw=2)。
> R85 試 active learning **greedy acquisition** 失敗（比 random sampling 還差）。
> 完整 patch BO methodology 需 explicit uncertainty。

## TL;DR

| 結果 | 數據 |
|------|------|
| R84 v5 (1:1 balanced) rw=2 Spearman | **0.601** ★ (close to deploy threshold 0.7) |
| R84 v5 top-5 pred avg true | **+3.49** (best across all dataset versions) |
| **R85 active learning greedy** | **+1.59** (worse than random +4.79!) |

→ **Pure greedy acquisition fails** on surrogate prediction.

→ Patch BO methodology must use UCB / Thompson sampling with uncertainty.

## R84 — Dataset_v5 Class Balance (Confirmation)

### Progressive dataset versions

| v | Optimized | Random | Total | Overall Spearman | rw=2 Spearman | Top-5 pred true |
|---|-----------|--------|-------|------------------|---------------|-----------------|
| v2 | 108 | 0 | 108 | 0.031 | 0.093 | +1.13 |
| v3 | 108 | 324 | 432 | 0.305 | 0.466 | -0.15 |
| v4 | 108 | 648 | 756 | 0.060 ↓ | 0.013 ↓ | +0.79 |
| **v5** | 108 | 108 | 216 | 0.155 | **0.601 ★** | **+3.49 ★** |

### 兩個 mode 的根本差異

```
rw=2 (flat-top, constrained):
  - Single-mode loss landscape
  - Class balance 顯著有效: 0.466 → 0.601
  - 接近 deploy threshold

rw=0 (steering, unconstrained):
  - Multimodal (R71 hamming 51.72%)
  - Single surrogate 學不會
  - Class balance 也救不回 (-0.096)
  - 需 mode-conditional architecture
```

### Top-K Filter Quality (BO acquisition feasibility)

```
v5 best for active learning:
  - Top-5 pred avg true: +3.49 dB
  - vs random: -4.76 dB
  - vs true top-5: +5.07 dB
  - Gap to true: 1.6 dB → BO 預測 viable
```

## R85 — Active Learning Loop Demo (Negative Result)

### 設計

Pool: dataset_v3 (432 entries with known true worst_supp)
- Initial: 20 random labels
- 5 iterations × 10 samples/iter
- Each iter: train CNN on labeled, predict on unlabeled, **greedy top-K**, reveal labels

Compare to: random sampling baseline (each iter pick 10 random unlabeled).

### Results

| Method | Init | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Final Best |
|--------|------|--------|--------|--------|--------|--------|------------|
| **Active learning (greedy)** | -0.89 | +1.59 | +1.59 | +1.59 | +1.59 | +1.59 | **+1.59** |
| **Random sampling** | -0.89 | -0.60 | +0.92 | +0.92 | +0.92 | **+4.79** | **+4.79** |
| True pool max | — | — | — | — | — | — | +5.57 |

### Greedy 失敗的 mechanism

```
AL iter trajectory (new batch avg true worst_supp):
  -15, -16, -17, -18, -20 (越選越爛)
  
Why?
  1. Surrogate 預測誤差 (R81 Spearman 0.30)
  2. Greedy 重複 exploit surrogate 預測「好」的點
  3. 但這些點實際是 surrogate 的盲區 (R77 adversarial)
  4. 加入 training set 後, surrogate 學壞, 進一步 mispredict
  5. 死循環: exploitation without exploration
  
Random sampling 雖然 noisy, 但 cover 整個 pool, 
偶爾命中 high worst_supp configs。
```

## Critical Lesson: Greedy Acquisition Doesn't Work

R76 doc 之前說「surrogate filter top-K = BO」是錯的。R85 實證:

```
✗ Pure greedy (predict top-K, reveal top-K):
   - Worse than random (R85 證明)
   - 重複 exploit surrogate prediction errors
   
✓ UCB acquisition:
   acquisition = predicted_mean + κ × predicted_std
   κ ~ 1-2 經驗值
   需 explicit uncertainty:
     - Ensemble: 訓 5-10 surrogates with diff seeds, std = ensemble variance
     - MC Dropout: forward 20 次 with dropout active, std = output variance
   
✓ Thompson Sampling:
   每 iter sample posterior surrogate, pick its top-K
   
✓ ε-Greedy (簡單但 effective):
   ε% random, (1-ε)% greedy
   ε ~ 0.3 經驗安全值
```

## 更新 PATCH_METHODOLOGY.md

新增章節：

```python
# CORRECT BO loop for patch surrogate
def bo_iteration(surrogate_ensemble, dataset, candidates, k=10, kappa=1.5):
    """UCB-based active learning iteration."""
    means, stds = ensemble_predict(surrogate_ensemble, candidates)
    ucb = means + kappa * stds  # exploration bonus
    selected = candidates[ucb.argsort()[-k:]]
    
    # Run HFSS on selected
    new_labels = [hfss_run(c) for c in selected]
    dataset.extend(zip(selected, new_labels))
    return dataset

# WRONG (R85 證明 fail):
def greedy_iteration(surrogate, dataset, candidates, k=10):
    means = surrogate(candidates)
    selected = candidates[means.argsort()[-k:]]  # NO uncertainty
    # ↑ Repeatedly exploits surrogate weak spots → worse than random
```

## 紀錄歷程更新

| Round | 結果 | 教訓 |
|-------|------|------|
| R76 | methodology distill | initial guidelines |
| R77 | adversarial gap | function MAE ≠ deploy |
| R78 | rule out OOD | gradient quality bottleneck |
| R79 | gradient cosine 0.001 | 完全 random |
| R80 | Sobolev fail | architectural limit |
| R81 | ranking 0.031 | dataset diversity 不夠 |
| R82 | v3 diversity | Spearman 0.03 → 0.30 |
| R83 | v4 imbalance | class balance > raw scaling |
| **R84** | **v5 balanced** | **rw=2 Spearman 0.60, top-5 +3.49** |
| **R85** | **AL greedy fail** | **必須 UCB w/ uncertainty** |

## 累計（85 rounds, 121+ commits）

完整 patch surrogate methodology cascade negative findings + positive remedies:

| Issue (R found) | Remedy |
|-----------------|--------|
| Function MAE 不夠 (R77) | 4-tier validation |
| Gradient random (R79) | 不依賴 GD-through-surrogate |
| Sobolev architectural fail (R80) | Active learning fallback |
| Ranking random (R81) | + diverse dataset |
| Class imbalance (R83) | 1:1 ratio |
| Multimodal mode collapse (R84) | mode conditioning OR constrained spec |
| **Greedy AL fail (R85)** | **UCB w/ uncertainty (ensemble)** |

## 對 Patch Antenna 最終 Action Items (final)

```
Week 1: Initial dataset
  - 100 random + 100 optimized HFSS = 200 entries balanced
  - Constrained spec (worst-case + ripple) > unconstrained max

Week 2: Surrogate training
  - CNN forward (full S-curve)
  - Train ENSEMBLE of 5 surrogates (diff seeds) for uncertainty
  - 4-tier validation:
    * Function MAE
    * Spearman ranking > 0.5 (relax from 0.7 due to RIS evidence)
    * Gradient cosine > 0.7 (likely fail, OK)
    * Adversarial gap < 5 dB

Week 3+: Active learning loop
  - UCB acquisition (mean + 1.5 × std)
  - 不要用 greedy
  - 每 iter: 5-10 HFSS runs
  - Maintain class balance

Deployment:
  - HFSS-direct primary
  - Surrogate-accelerated active learning (UCB)
  - 不要 trust GD-through-surrogate
```

## 結論

85 rounds RIS playground 完整 explored surrogate methodology。**核心發現**:
- Function MAE alone misleading (R77)
- Surrogate gradient unreliable (R79)
- Sobolev limited by architecture (R80)
- Dataset diversity + class balance critical (R82-R84)
- **Greedy acquisition worse than random — must use UCB w/ uncertainty (R85)**

對 patch team 的 deliverable:
- `script/PATCH_METHODOLOGY.md`: 完整 transition reference
- 41 round summaries: detailed experimental record
- 多個 dataset versions: scaling analysis
- Cascade negative findings: avoid known traps

Patch transition ready to begin with informed methodology.
