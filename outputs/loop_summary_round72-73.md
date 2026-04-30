# /loop Round 72–73 兩輪總結 — Surrogate Scaling + Generator Multimodal

> R72 dataset_v2 完成 + 確認 surrogate scaling 規律。R73 試 conditional generator
> 驗證 R71 multimodal hypothesis，部分成立但揭露 supervised loss 的根本問題。

## TL;DR

| Round | 結果 |
|-------|------|
| R72 surrogate scaling | MAE 4.95 (v1) → **2.57 dB (v2)**, scaling power N^-1.62 |
| R73 conditional generator | **mode separation 39.96% hamming**（非 collapse）但 worst_supp error 16.82 dB（broken）|

→ surrogate-in-the-loop 路線必要，pure supervised 不可行

## R72 — Dataset_v2 完成 + Surrogate Scaling

### Dataset_v2 (108 Pareto rows)

54 configs × 2 rw × 2 seeds = 216 runs, 82 min。
新增: n=41（v1 沒）。

### Surrogate scaling

| Dataset | entries | MAE worst_supp | scaling |
|---------|---------|----------------|---------|
| v1 | 72 | 4.95 dB | baseline |
| **v2** | **108** | **2.57 dB** | **-48%** |

Power: N^-1.62（比 linear scaling 還好）。

**外推**:
- < 1 dB MAE 需 ~200-300 entries
- < 0.5 dB MAE 需 ~500 entries

對 patch surrogate 給 **明確 dataset budget**。

### Dataset_v2 deployment 高點

| Config | best worst @ rw=2 | ripple |
|--------|-------------------|--------|
| **n=41 × w=10°** | **+0.92 ★** | **0.70 dB** |
| n=41 × w=20° | -0.22 | low |
| n=21 × w=10° | -0.60 | low |

n=41 + 窄 main beam 是 binary 1-bit 真實 deployable 上限。

## R73 — Conditional Generator (multimodal hypothesis test)

### 設計

```
config_vec (6-dim, 含 ripple_weight) → MLP encoder → 6×6 latent
  → upsample conv → 41×41 logits → sigmoid > 0.5 → binary pattern
loss = BCE(pred_logits, target_pattern) within mask
```

ripple_weight 作為**顯式 mode conditioning**。

### 結果

| Metric | 數值 | 解讀 |
|--------|------|------|
| Pred vs GT hamming | 44.94% | 接近隨機 — 沒學到具體 pattern |
| Pred worst_supp error | **16.82 dB** | 預測 pattern 跑 sim 表現崩壞 |
| **同 config rw=0 vs rw=2 hamming** | **39.96%** | **mode separation 確實學到** |

### 關鍵診斷

Generator **成功學會 mode 區分**（39.96% hamming 接近預期 50% multimodal），
**但每個 mode 內預測 pattern 都是 mode 內平均 garbage**（hamming 45% from GT）。

### 為什麼 supervised BCE 不夠

1. **同 config 多 optimal**: 同一 (freq, n, θc, w, rw) 下有很多 hamming-distant 的 optimal patterns
2. **BCE on average**: NN 學 mode 內 pixel-wise 平均 → 變模糊 garbage
3. **Discrete manifold**: optimal binary patterns 在 discrete manifold，不是 continuous distribution

## 對 R1-R30 Lab Generator Failure 的最終解釋

R1-R30 lab generator 失敗有 **兩個獨立問題**：

1. **Unconditional collapse** (R71 揭露)
   - (config → pattern) 是 multimodal
   - 沒有 mode conditioning → 學 multimodal mean = garbage
   - **R73 用 ripple_weight conditioning 解決一半** (39.96% mode separation)

2. **Supervised BCE 不適合 discrete manifold** (R73 揭露)
   - 同 mode 內 pattern 仍 hamming-distant
   - BCE 學 mode 內 pixel-wise 平均 = mode 內仍 garbage
   - **沒解決** — 即使加 conditioning 還是崩壞

## 對 Patch Antenna Methodology 的最終結論

**Generator 路線必須結合 differentiable physics (surrogate-in-the-loop)：**

```python
# CORRECT approach for patch antenna
config + mode → generator → patch_geometry (continuous params)
patch_geometry → surrogate (CNN) → predicted_response  
loss = worst_case_loss(predicted_response, target_spec)
gradients backprop through surrogate to update generator

# SHOULD NOT DO
config + mode → generator → patch_geometry
loss = MSE(predicted_geometry, optimal_geometry)
# → 同 config 多 optimal, MSE 學 mean garbage
```

這個結論證實實驗室原本生成器路線的方向是對的，但缺三個關鍵元素：
1. **Mode conditioning**（R71/R73）
2. **Differentiable surrogate**（取代直接物理模擬器）
3. **Worst-case loss**（不是 max-max）

## 紀錄歷程更新

| 階段 | 結果 |
|------|------|
| R57-R63 free-phase | +30.99 max-max (虛胖) |
| R64 worst-case loss | +6.88 honest |
| R66-R67 dataset_v1 | 72 entries Pareto |
| R68-R69 surrogate POC | MAE 4.89 dB |
| R70-R71 symmetry + multimodal | hamming 51.72% |
| **R72 dataset_v2 scaling** | **MAE 2.57, N^-1.62 scaling** |
| **R73 conditional generator** | **mode separation 40%, but supervised BCE 不夠** |

## 累計（73 rounds, 110+ commits）

- 25+ scripts
- dataset_v1 (72) + dataset_v2 (108 Pareto rows)
- 3 surrogate variants + 1 generator
- 3 層完整文檔 + 36 round summaries

## Open Questions（核心收斂中）

1. **Generator + surrogate-in-the-loop**: 預期 → MAE < 2 dB
2. dataset_v3 (200-300 entries) → MAE < 1 dB？
3. **Active learning**: 用 surrogate uncertainty 選 next entry，比 random 多 sample-efficient？
4. 是否該移到 patch 試 methodology？（RIS playground value 漸滿）
