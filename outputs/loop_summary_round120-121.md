# /loop Round 120–121 兩輪總結 — Sidelobe Area Minimization Champion Recipe

> R118 揭露 mean(side) penalty 是壓低 sidelobe area 的關鍵。
> R119 grid search 找到 (rw=2, λ=1) sweet spot。
> R120 視覺證明 main 整片貼上蓋 + sidelobe 整體壓低同時達成。
> R121 stack with multi-bit phase, 找到 CHAMPION recipe (2-bit + λ=1)。

## TL;DR

**從 R94 baseline 到 R121 CHAMPION：side_mean 整片下降 15 dB，main 始終貼上蓋**

| Recipe | worst | side_max | side_mean | flat-top |
|--------|-------|----------|-----------|----------|
| R94 baseline (1-bit, λ=0) | +1.92 | -4.51 | -15.75 | ✓ |
| R119 (1-bit, λ=1) | +3.65 | -6.60 | -23.70 | ✓ |
| **R121 CHAMPION (2-bit, λ=1)** | **+3.45** | **-6.05** | **-30.84 ★** | **✓** |

對應原始 spec：
- ✓ Main beam region 整片接近上蓋（5/5 flat-top, all 3 recipes）
- ✓ Sidelobe 整體越低越好（-15 dB total mean shift across 2 stacks）

## R120 — Visual Proof of R119 Recipe

`outputs/r120_baseline_vs_winner.png` side-by-side:
- R94 vs R119
- 兩者都 main 整片貼上蓋（0/30 violations）
- R119 sidelobe distribution 顯著左移 -8 dB

驗證：mean penalty 同時：
- 維持 main flat-top compliance
- 大幅壓低整片 sidelobe distribution

## R121 — Multi-bit Stacking Champion

完整 (bits × λ_mean) grid 揭露 stack effect:

```
1-bit + λ=1: side_mean -23.70 ✓ (R119)
2-bit + λ=1: side_mean -30.84 ✓ ★ NEW CHAMPION
3-bit + λ=1: side_mean -31.01 (saturated, marginal)
1-bit + λ=3: side_mean -30.46 ✗ lose flat-top
```

### 關鍵 insight

```
λ_mean=1 是 universal flat-top boundary:
  λ ≤ 1: 100% flat-top (across all bits)
  λ ≥ 2: 0% flat-top (regardless of bits)

Multi-bit phase + mean penalty 真實 stack:
  1-bit baseline:  -15.75
  + mean penalty:  -23.70  (-7.95 from λ=1)
  + 2-bit phase:   -30.84  (-7.14 from 2-bit)
                   ════
                   -15.09 total
                   
3-bit saturate: -31.01 vs -30.84 (僅 +0.17, 不值得 3-bit cost)
```

## R122 — Visual: 三 Recipe Progression

`outputs/r122_three_recipes.png` shows:
- Row 1 R94: side spread -10 ~ -40, mean -15.75
- Row 2 R119: side shifted lower, mean -23.70
- Row 3 R121: side tight cluster -30 ~ -40, mean -30.84
- 三層 strict 改善 + main 始終貼上蓋

## Final Production Recipes Tiered

```python
# Tier 1: Cost-sensitive (1-bit RIS-style hardware)
recipe_economy = {
    "phase_resolution": "1-bit binary",
    "ripple_weight": 2.0,
    "mean_lambda": 1.0,                     # ← R119 NEW
    "expected": {
        "worst": "+3.65 dB",
        "side_mean": "-23.70 dB",
        "flat_top": "100%",
    }
}

# Tier 2: Production sweet (2-bit phase shifters) ★
recipe_production = {
    "phase_resolution": "2-bit (4 levels)",
    "ripple_weight": 2.0,
    "mean_lambda": 1.0,
    "expected": {
        "worst": "+3.45 dB",
        "side_mean": "-30.84 dB ★",
        "flat_top": "100%",
    }
}

# Tier 3: Premium (3-bit, marginal)
recipe_premium = {
    "phase_resolution": "3-bit (8 levels)",
    "expected": {
        "side_mean": "-31.01 dB (only +0.17 vs 2-bit)",
        "verdict": "Not worth 3-bit cost",
    }
}
```

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R118 | mean(side) penalty 探索, λ=0.3 +mean shift -8 dB |
| R119 | Grid 找 sweet spot: rw=2 + λ=1, strict improvement |
| R120 | Visual proof R94 vs R119 |
| R121 | Multi-bit stack: 2-bit + λ=1 CHAMPION |
| R122 | Three-recipe progression visual |

## 對 patch transition 的最終 update

之前 (R94 baseline) recipe:
```
loss = -(min(main) - max(side)) + 2.0 * (max(main) - min(main))
→ worst +1.92, side_mean -15.75
```

現在 (R121 CHAMPION) recipe:
```
loss = -(min(main) - max(side))
     + 2.0 * (max(main) - min(main))
     + 1.0 * mean(side)              ← NEW (R119)
+ 2-bit phase quantization           ← STACK (R121)
→ worst +3.45, side_mean -30.84
```

對應 patch hardware:
- 2-bit phase shifters: production sweet spot
- λ_mean=1.0: universal flat-top boundary
- Total side_mean improvement: -15 dB vs baseline

## 累計 (122 rounds, 158+ commits)

**Methodology 從只看 worst → 整片 distribution shaping：**
- R64: worst-case loss (worst sidelobe down)
- R94: + ripple penalty (main flat)
- R119 NEW: + mean penalty (整片 distribution down)
- R121 NEW: + multi-bit phase (stacking)

完整 production recipe 達到 side_mean -30 dB + flat-top 100%。

下一階段：cross-axis robustness 驗證 R121 champion 在 inc/freq/n 變化下是否仍保持。
