# /loop Round 76–77 兩輪總結 — Methodology Distill + Critical Validation Test

> R76 把 75 rounds 經驗精煉成 patch methodology doc。R77 直接 self-validate：
> 用 R72 surrogate 做 GD optimization 跟 real-sim baseline 比，**揭露關鍵 missing step**。

## TL;DR

**R77 critical finding: surrogate test MAE 2.57 dB 不夠 deploy。GD 透過 surrogate 比
real-sim 差 7-25 dB worst_supp（adversarial overfitting）**

→ **R76 methodology doc 必須加上 surrogate validation protocol**（已更新）

## R76 — `script/PATCH_METHODOLOGY.md`（distilled）

75 rounds 經驗精煉成 patch transition reference:

### 5 條核心原則
1. Loss = use case，不是 abstract metric
2. Dense supervision >> sparse supervision (R68/R69)
3. CNN spatial inductive bias 必要 (R68)
4. Multi-restart 5+ seeds (R44/R56)
5. 識別 infeasible region (R67)

### 4 個 Architecture Templates
- Forward Surrogate
- Per-Target Optimization
- Generator E2E (with surrogate-in-loop, NOT supervised BCE)
- Worst-Case Loss

### Dataset Schema + Pareto Frontier
- 多 ripple weight per entry
- Multi-seed
- Class balance + edge cases

### Transferable vs RIS-only 區分

## R77 — Surrogate-in-loop GD vs Real-sim GD

### 設計

對 5 個 test targets，跑兩套 optimization:
- A. Real-sim GD (R64 baseline): GD through differentiable RIS sim
- B. Surrogate-in-loop GD (R76 methodology): GD through frozen R72 surrogate

兩個 final pattern 都 eval through real sim。

### 結果

| Target | Real-sim worst | Surrogate worst | Gap |
|--------|----------------|-----------------|-----|
| 38G n=31 0° w=20 rw=2 | -1.55 | -8.55 | **+7.00** |
| 38G n=31 -30° w=10 rw=2 | -0.89 | -15.46 | **+14.57** |
| 28G n=31 0° w=20 rw=2 | -0.32 | -7.80 | **+7.48** |
| 38G n=41 0° w=10 rw=2 | +1.45 | -12.58 | **+14.03** |
| 38G n=31 30° w=30 rw=0 | -0.98 | -25.61 | **+24.63** |

**Mean gap: +13.54 dB**（surrogate-in-loop 比 real-sim 差非常多）

### 為什麼失敗：Surrogate Exploitation

```
Surrogate test set MAE = 2.57 dB（隨機 sample 統計）
              ↓
              ↓ GD 不是隨機 sample，主動沿 gradient 方向
              ↓ 找 surrogate 預測誤差**最大化**的點
              ↓ 即 GD 找到 surrogate 弱點
              ↓
GD-found pattern: surrogate 預測「很好」, real sim 預測「很爛」
              ↓
Gap = 7-25 dB
```

技術解讀:
1. Surrogate trained on binary {0, 1} patterns
2. GD parametrize 連續 free-phase, soft_bin = sigmoid(...) 中間態
3. soft_bin during early training 在 0.5 附近 → OOD for surrogate
4. Surrogate gives unreliable gradients in OOD region
5. GD walks toward whatever direction OOD garbage points
6. Result: pattern 在 surrogate 視角好，但 real sim 完全不同預測

### Critical Lesson 給 patch methodology

**不能只用 random test set MAE 驗證 surrogate**。GD trajectory ≠ random sample。

新增 mandatory validation protocol（已寫入 PATCH_METHODOLOGY.md）:

```python
# 1. Surrogate test MAE < 1 dB ✓ (necessary)
# 2. Adversarial validation: 
#    surrogate-loop opt vs HFSS-loop opt on 10 diverse targets
#    Gap < 2 dB → OK to deploy
#    Gap 2-5 dB → 加 uncertainty/ensemble
#    Gap > 5 dB → 不可 deploy, 擴 dataset
# 3. Active learning loop if gap > 2 dB
```

### 對 patch 的 implication

Patch surrogate 可能比 RIS 好（continuous geometry, smoother prediction），但
adversarial gap 仍會存在。**必須驗證，不能假設 random test MAE 等於
deployment quality。**

## 紀錄歷程更新

| 階段 | 結果 | 教訓 |
|------|------|------|
| R57-R63 | +30.99 max-max | 虛胖 metric |
| R64 | +6.88 worst-case | metric 必須 reflect physics |
| R66-R72 | dataset + surrogate scaling | N^-1.62 power |
| R71 | hamming 51.72% | (config → pattern) multimodal |
| R73-R75 | generator failures | binary + supervised 不行 |
| R76 | methodology distilled | 5 原則 + 4 templates |
| **R77** | **surrogate-loop adversarial** | **MAE 不等於 deploy quality** |

## 累計（77 rounds, 114+ commits）

- 28+ scripts
- 2 datasets, 4 surrogate variants, 2 generator variants
- methodology doc with adversarial validation protocol
- 38 round summaries

## 下一階段建議（更新）

1. **Patch transition (still recommended)**: 帶上 R77 的 adversarial validation。
   不要直接相信 surrogate MAE。

2. **OOD-robust surrogate training**: R77 之外，R78+ 可試 augment dataset_v2 with
   continuous interpolated patterns 看能否關 adversarial gap。

3. **Active learning**: HFSS-verify surrogate-loop outputs, add bad ones to training。
   經典 active learning loop。

4. **Ensemble surrogates**: 訓多個 surrogate, variance 當 uncertainty proxy,
   reject patterns where ensemble 不一致。

我建議 (1) + (3): patch transition with 內建 active learning loop。
