# /loop Round 57–58 兩輪總結 — 重大演算法突破

> 接續 round 55-56 總結。Round 57 意外發現 free-phase parameterization
> 突破之前以為的 binary ceiling +13.44 dB。Round 58 確認 universality。

## TL;DR — NEW RECORD: +21.31 dB（破舊紀錄 +7.87 dB）

| 路線 | 5.6 GHz × 19 | 28 GHz × 13 |
|------|--------------|-------------|
| 舊（sigmoid+SA） | +11.82 (R37) | +13.44 (R47) |
| **新（free-phase）** | **+19.61 (R58)** | **+21.31 (R57) ★** |
| Δ | +7.79 dB | +7.87 dB |

從 v1 (-4.08) 到 R57 (+21.31) = **+25.39 dB 累計改善**

## Round 57 — 演算法突破

### 動機

R55 文獻 connection 指出 1-bit RIS 有 ~3 dB quantization loss vs continuous
phase。為實證此 gap，做 continuous vs binary 對照——意外發現舊 sigmoid 路線
受限於半圓相位。

### 三個關鍵 algorithmic 改變

1. **Free-phase parameterization**
   - 舊：`logits → sigmoid ∈ [0,1] → phase ∈ [0, π]`（半圓）
   - 新：`params ∈ ℝ → phase ∈ [0, 2π)`（全圓，no constraint）

2. **Direct logsumexp loss**（取代 tolerance loss）
   ```python
   main_soft = (1/β) * logsumexp(β * resp[main_mask])
   side_soft = (1/β) * logsumexp(β * resp[~main_mask])
   loss = -(main_soft - side_soft)  # 直接最大化 suppression
   ```
   - 舊 tolerance loss 在 `side ≤ -25` 後 saturate, gradient 死掉
   - 新 direct loss 永遠提供 gradient 信號

3. **Optimal 1-bit quantization**（基於 phase 在單位圓上距離）
   ```python
   phase = (params * π) % (2π)  # in [0, 2π)
   bin = ((phase > π/2) & (phase < 3π/2)).float()  # closest to {0, π}
   ```

### 30-Seed 結果（28 GHz × 13 × +51° × width=80）

| Metric | Free continuous | 1-bit quantize | + SA |
|--------|-----------------|----------------|------|
| Mean | +30.50 | +15.72 | +15.89 |
| **Max** | **+34.70 (seed 26)** | **+21.31 (seed 4) ★** | **+21.31** |
| Min | +26.90 | +11.37 | +11.37 |

**Top 5 seeds**: seed 4 (+21.31), 19 (+20.44), 18 (+19.99), 7 (+18.35), 27 (+17.89)

SA 對 free-phase 結果只小幅改善（+0~+2 dB），因 free-phase GD 已收斂到
deeper basin。

### Quantization Loss 實證

free continuous mean +30.50 → 1-bit mean +15.72 = **gap ~+14.8 dB**

這比文獻「3 dB beam-gain loss」大很多。原因：
- 文獻 3 dB 是 main beam **peak gain** 損失
- 我們的 metric 是 **suppression**（main - side），對 phase precision 更敏感
- Suppression 跟 null depth 直接相關，binary phase 無法做精細 cancellation

## Round 58 — Universality Test

5.6 GHz × 19 × +60° × width=46 × broadside × 10 seeds × free-phase+SA：

| Best seed | Result |
|-----------|--------|
| seed=6 + SA | **+19.61 dB ★** |
| seed=9 raw | +18.94 dB |
| mean | +17.31 dB |

vs R37 舊紀錄 **+11.82**：**+7.79 dB 改善**（跟 R57 +7.87 dB 一致）

→ ~+7.8 dB universal improvement 確認

## 為什麼舊路線（sigmoid）這麼差？

1. **半圓相位限制**：sigmoid → phase ∈ [0, π]，失去一半相位自由度
2. **Post-quantize >0.5 不是 optimal**：直接套半圓中點
3. **Tolerance loss saturate**：side ≤ -25 後 zero gradient
4. **Continuous space 卡 ~+4.85 dB**（R57 sigmoid mean）
5. SA 雖能 explore 部分 binary space，但起點是 sigmoid 卡的 local optimum

free-phase + direct loss 同時解決這 4 個問題。

## Epistemic 反轉

**之前 R47-R56 9 rounds 嘗試突破 +13.44**：
- inc fine grid (R48 knife-edge ±1°)
- plateau pos sweep (R52)
- freq fine grid (R49/50/53/54)
- width fine grid (R55)
- seeds 0-9 (R56a)
- larger n (R51/56b)

全部失敗 → 結論「+13.44 是 binary ceiling」**錯誤**。

**真實 ceiling 在演算法層面，不在 hyperparameter sweep 層面**。
+13.44 是 sigmoid path 的 attraction basin upper bound，不是 binary
物理上限。

### 啟示

> **演算法選擇比 hyperparameter sweep 重要 10×**

文獻 search 不是 exhaustively 抄 paper，是用 paper 找出當前路線的 blind spot。
R55 文獻 connection 提示「continuous vs binary」對照，做這個對照才發現
舊路線根本沒在做合理的 continuous 優化。

## 累計（58 rounds, 92+ commits）

### 工具庫
20+ scripts:
- 3 design tools (sigmoid path)
- 2 NEW: continuous_vs_binary_eval.py, verify_free_phase_record.py (free-phase path)
- 6 sweep tools
- 4+ benchmark tools

### 完整 epistemic 鏈
- R1-R30: explore generator path → conditioning failure
- R30-R56: sigmoid GD + SA 路線優化
- **R57-R58: free-phase 演算法替換 → +7.8 dB universal improvement**

## Open Questions（更新）

1. Free-phase + multi-restart with SA-per-restart 能否再突破 +21.31？
2. Free-phase 在 60 GHz / 38 GHz 等其他 freq 是否同樣 +7.8 dB 改善？
3. Continuous Re/Im parameterization 是否再優於 phase parameterization？
4. Free-phase ceiling 是否就是 1-bit 物理上限？(continuous +30 vs binary +20)
5. 是否能設計 generator 學「給 random pre-phase 然後做 free-phase GD」？
   → 部分回到 lab generator 思路但 grounded in 真實 1-bit physics
