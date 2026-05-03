# /loop Round 144–145 兩輪總結 — Surrogate scaffolding 連續負面結果

## TL;DR

R144 用 trajectory snapshots 重訓 surrogate（修 R142/R143 的 distribution mismatch）→
**比隨機資料更糟**（in-dist R²=-3.2 vs OOD R²=-1.5）。R145 嘗試從 analytical sim 直接
warm-start 權重 → **untrained R²=-0.97**，事後 diagnosis 發現是兩個權重提取 bug
（phi axis index 錯 + element flatten ordering 與 sim 的 `.t()` 不一致）。

兩輪連續負面結果。Phase 2 surrogate scaffolding 比預期難很多。**好消息**：
都在 RIS playground 階段發現，沒燒掉 patch surrogate 的時間。

## R144 — Trajectory-Distribution 重訓

### 設計

- 跑 15 次 R141 optimization (R119 recipe at n=31, inc=51, 38GHz, w=10)
- 每 30 步 snapshot quantized binary pattern + response → 50 snapshots/run
- 12 train seeds (600 patterns) + 3 test seeds (150 patterns) + 300 random OOD

### 預期 vs 實際

| 預期 | 實際 |
|------|------|
| In-dist trajectory data → 高 R² | **R² = -3.2 (更糟!)** |
| OOD random → 較低 R² | R² = -1.5 |

### Diagnosis

Trajectory snapshots 跨 50 dB dynamic range（main 接近 0 dB, sidelobes -40+ dB），
random patterns 多在 -10 to -30 dB 中段。Surrogate 從 random init 學「平均響應」
（≈ -20 dB）對 trajectory 的 extreme values 預測誤差爆大，反而對 random 比較準
（因為 random 大多就在中段）。

```
Trajectory data variance:  HUGE (50 dB span)
Random data variance:      MEDIUM (20 dB span)
Model predicts ~constant:  errors scale with target variance
→ trajectory R² explosive negative
```

訓練 loss 確實有降（334 → 140 over 100 epochs）但 test 卡住（357 → 320），
train/test gap 200 → 嚴重 overfit on 600 samples / 695K params。

## R145 — Warm-Start from Analytical Coefficients

### 想法

Analytical sim 結構：
```python
af = pre_calAF * exp(j * pattern * pi)        # complex (n_phi, n_theta, n²)
AF = |sum_{i,j} af|                          # (n_phi, n_theta)
response = 20 * log10(AF / max(AF along theta))  # 標準化到峰值
```

理論上 surrogate 的 (real_lin, imag_lin) weights 就是 Re/Im(pre_calAF) 在
broadside phi 切片。直接複製進去 → 應該無需訓練即達到 R² ≈ 1。

### 結果

```
Untrained warm-start fit:
  R^2:                        -0.9653
  mean |abs err|:             6.09 dB
  max  |abs err|:             55.6 dB
```

**完全不對**。

### Diagnosis (post-mortem)

仔細重讀 `antenna/ris/simulate_ris.py` 後發現兩個 bug：

| Bug | 描述 | Fix |
|-----|------|-----|
| 1 | 用 `phi_idx=45` (phi=90°) | 應該 `phi_idx=0` (sim 返回 `dB_AF[0]` = phi 第 0 切片 = 0°) |
| 2 | `x.flatten()` 直接展平 | sim 是 `MPD.t().reshape(...)` — 我沒做 `.t()`，element ordering 不一致 |

Architecture 本身是對的（sim 確實是 `|W·x|² + log` 結構），只是 weight 提取
實作錯誤。

## Phase 2 階段觀察

連續 4 輪嘗試：

| Round | Approach | R² | 備註 |
|-------|----------|------|------|
| R142 | 標準 CNN + random data | ≈ 0 | stuck on mean |
| R143 | Physics-aware + random data | -0.74 | overfit |
| R144 | Physics-aware + trajectory data | -3.21 | 反而更差 |
| R145 | Warm-start from analytical | -0.97 | 提取 bug |

每輪都有不同 failure mode，但都指向同個結論：**surrogate-in-the-loop 的數據工程
+ 架構設計 + 初始化 都需要 careful execution**，cold-start 從 random init
很難找到正確 weight manifold。

對 patch transition 的 implications：
- HFSS data 是 expensive 的 — 這些失敗如果發生在 patch 階段就是大量 GPU/CPU 時間
- Surrogate 需要 careful curation：trajectory data + random data 混合 + warm-start
- 現有 SurrogateModel + HFSSNet 在 codebase（`antenna/models/surrogates/`）已是
  比較成熟的設計，未來 patch 階段應參考其架構選擇

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R144 | Trajectory data 反而更糟（response variance 太大）|
| R145 | Warm-start 提取 bug → 兩個 indexing 錯誤 |

## 下一階段建議 (R146+)

優先順序：

1. **R146**: 修 R145 的兩個 bug + verify warm-start untrained R² = 1.0
   （這是 sanity check，證明 architecture 是對的）

2. **R147**: 用修好的 warm-start surrogate 跑 surrogate-loop optimizer，
   驗證 R141 recipe 在 surrogate gradient 下還產生好 pattern

3. **R148**: 隨機 perturb warm-start 權重 (~5% noise) 模擬 patch HFSS surrogate 的
   不完美，驗證 robustness

4. **降級備案 (R149)**: 若 surrogate-loop 始終 unstable，改用 CMA-ES gradient-free
   optimizer + analytical sim 直接 patch transition

## 結論

Phase 2 連續 4 輪 negative results，但每輪都精準 surface 不同 failure mode。
這正是 RIS playground 的價值 — 在 patch 之前 derisk 所有 methodology bug。

R141 的 deployment API 仍 production-ready 對 analytical sim，沒有 regression。
Phase 2 的進度比預期慢，但 lessons learned 都會直接 transfer 到 patch surrogate
階段，避免重蹈覆轍。
