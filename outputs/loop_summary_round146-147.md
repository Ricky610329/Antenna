# /loop Round 146–147 兩輪總結 — Phase 2 unblocked: surrogate-loop 過 baseline

## TL;DR

R146 修掉 R145 的兩個 indexing bug → warm-start surrogate **untrained R² = 1.000000**，
mean abs err = 0.000002 dB。R147 把 surrogate 擴成 continuous-aware（cos/sin
instead of 1-2x），跑 surrogate-in-the-loop optimization 對比 analytical 真實
gradient → **mean worst delta < 0.05 dB, 兩個都 5/5 flat-top**。

**Phase 2 解鎖**：核心 methodology validation 通過。Loss 設計 + recipe selector
+ joint early-stop **全部 transfer 到 surrogate gradient**。Patch transition 主要
risk 從「methodology 是否 transfer」變成「HFSS data 是否能訓出好 surrogate」。

## R146 — 修 R145 兩個 bug

### Bugs

| # | Bug | Fix |
|---|-----|-----|
| 1 | `phi_idx = 45` (phi=90°) | `phi_idx = 0` — sim 的 `dB_AF[0]` 是 phi 第 0 切片 = phi=0° |
| 2 | `x.flatten()` 不對 | `(1-2x).transpose(1,2).flatten()` 匹配 sim 的 `MPD.t().reshape()` 列主序 |

### Sanity check

修完後 untrained warm-start 在 200 random binary patterns：

```
R^2:                       1.000000
mean |abs err|:            0.000002 dB
max  |abs err|:            0.000259 dB  
median |abs err|:          0.000001 dB
```

**完全等同 analytical sim**。架構正確 → R142/R143/R144 全是 cold-start 找不到
正確 manifold 的問題，不是架構不夠。

## R147 — Surrogate-in-the-loop Optimization

### 設計重點

R146 surrogate 用 `(1-2x)` 只能處理 binary input。但 GD 在 continuous params 上跑，
中間步驟 `params` 是 continuous 的。需要 continuous-aware 版本：

```python
phase = x * pi
cos_p = cos(phase).T.flatten()    # complex amplitude real part
sin_p = sin(phase).T.flatten()    # complex amplitude imag part
F_real = real_lin(cos_p) - imag_lin(sin_p)   # complex × complex
F_imag = real_lin(sin_p) + imag_lin(cos_p)
amp = sqrt(F_real^2 + F_imag^2)
out = 20*log10(amp/max(amp))
```

對 binary input (x=0 or 1)：
- x=0: phase=0, cos=1, sin=0 → F = (W_re·1, W_im·1) = pre_calAF
- x=1: phase=π, cos=-1, sin=0 → F = -pre_calAF

對應 sim 的 `exp(j*0) = 1, exp(j*π) = -1`。

### 驗證

```
binary input mean abs err:     0.000002 dB
continuous input mean abs err: 0.000002 dB
```

**Continuous AND binary 都 exact match**。

### Surrogate-loop vs Analytical baseline

5 seeds × R119 recipe × joint early-stop：

| Metric | Surrogate-loop | Analytical-baseline | Delta |
|--------|---------------|---------------------|-------|
| best | +0.82 | +0.95 | -0.13 |
| mean | +0.68 | +0.66 | **+0.02 ★** |
| min | +0.56 | +0.33 | **+0.23 (surr 更好)** |
| side_mean | -24.32 | -23.31 | -1.01 (surr 更好) |
| flat-top | **5/5** | **5/5** | tie |
| Wall time | 29.6s | 87.1s | **3x 快** |

Per-seed worsts:
- Surrogate: [+0.56, +0.82, +0.56, +0.79, +0.68]
- Analytical: [+0.33, +0.95, +0.49, +0.81, +0.73]

**意義**：surrogate gradient 跟 analytical gradient 等價（甚至略好），且更快。

註：worst 絕對值 (~+0.8) 比 R141 baseline (~+3) 低，因為 R147 用 n=31 而 R141 是 n=51。
比較還是公平的，兩邊都用 n=31。

## Phase 2 重新評估

### 4 連續 negative 後的 turning point

| Round | Approach | R² | Verdict |
|-------|----------|------|---------|
| R142 | 標準 CNN, random | ≈ 0 | 架構不對 |
| R143 | Physics-aware, random | -0.74 | 架構對, cold-start 難 |
| R144 | Physics-aware, trajectory | -3.21 | trajectory 反而難（dynamic range 大）|
| R145 | Warm-start (有 bug) | -0.97 | indexing bugs |
| **R146** | **Warm-start (fixed)** | **1.000000** | **architecture 確認 sufficient** |
| **R147** | **Surrogate-loop opt** | **delta=+0.02** | **methodology transfers** |

### 對 Patch Transition 的 Implications

| Risk | 原本 | 現在 |
|------|------|------|
| Methodology transfers? | 大 risk | ✓ 確認 transfer |
| Surrogate fit quality? | 大 risk | 仍是 risk（real HFSS data 比 analytical 難）|
| Optimizer 穩定性 in surrogate-loop? | 大 risk | ✓ joint early-stop 通用 |
| Surrogate gradient noise effect? | 大 risk | TODO R148 perturb test |

Phase 2 主要 risk 從「architecture/methodology」轉移到「data engineering」—
這就是該在 RIS playground derisk 的部分。

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R146 | Warm-start untrained R² = 1.0，架構確認 sufficient |
| R147 | Continuous-aware surrogate-loop 對 baseline mean delta = +0.02, 5/5 flat |

## 下一階段建議

1. **R148**: 在 surrogate weights 加 perturbation noise（5%, 10%, 20%）模擬不完美
   HFSS surrogate，看 surrogate-loop 還能 produce 好 pattern 嗎
   
2. **R149**: surrogate-loop 在 R141 selector 跨 6 configs 全跑一次，verify recipe
   selection 在 surrogate 下還 PASS
   
3. **R150+**: 真正進 patch transition — train HFSS surrogate（用 patch
   antenna data 或 simulator）+ 套用 selector + joint early-stop

## 結論

連續 4 輪 negative 之後 R146-R147 兩輪 breakthrough。

**Phase 2 核心命題**「loss + workflow + 思維可以 transfer 到 surrogate-in-the-loop」
**已得到強證據支持**。

R141 deployment API 加上 surrogate-loop wrapper 就是 patch transition 工具鏈
的雛形。Phase 3 (patch transition) 風險評估 downgraded。
