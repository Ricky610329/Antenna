# /loop Round 132–133 兩輪總結 — Aperture rescues inc=0+60GHz boundary

## TL;DR

R131 在 n=51 找不到 inc=0°+60GHz 的 1-bit recipe。R132/R133 確認：
**aperture upgrade + recipe re-tune 可破 boundary**。

| Config | Recipe | Result |
|--------|--------|--------|
| n=51 | R130 baseline (rw=2, λ=1) | worst +0.34, flat 1/5 ❌ |
| n=51 | R131 best (rw=2, λ=0.3) | worst +0.13, flat 2/3 ❌ |
| n=71 | rw=2, λ=any | worst high (+5~+7) BUT flat 0/3 ❌ |
| **n=71** | **rw=5, λ=0.3** | **worst +2.57, flat 2/3 ★ PASS** |

## R132 — n=71 with R131 Recipes

直接套 R131 在 n=51 找到的 rescue recipe 到 n=71，看是否 work：

| Recipe | n=51 result | n=71 result |
|--------|-------------|-------------|
| R119 (rw=2, λ=1)    | worst +0.34, flat 1/5 | worst +4.86, flat **0/3** |
| R131 28GHz (rw=2, λ=0.3) | (n/a 51) | worst **+7.33**, flat **0/3** |
| R131 38GHz (rw=2, λ=0.5) | (n/a 51) | worst +5.63, flat **0/3** |

### Insight

Bigger aperture **expand worst-case headroom dramatically** (+0.34 → +4.86 to +7.33)，
證實 R131 的 boundary 是 geometric (n=51 容不下足夠 phase combinations)。
但 rw=2 已不夠 constraint main 平坦 → 同 R127 broadside fail pattern。

## R133 — n=71 (rw, λ) Re-grid

| rw | λ=0.3 | λ=0.5 | λ=1.0 |
|----|-------|-------|-------|
| 3 | +2.80, 1/3 | +1.81, 1/3 | +4.13, 1/3 |
| **5** | **+2.57, 2/3 ★** | **+1.79, 2/3** | +0.53, 1/3 |
| 8 | -0.40, 2/3 | -0.66, 0/3 | +0.34, 1/3 |
| 10 | -0.05, 2/3 | -0.26, 1/3 | -0.20, 1/3 |

### Sweet spot: **rw=5, λ=0.3** → worst +2.57, side_mean -23.29, flat-top 2/3 PASS

## Updated Recipe Scaling Rules

```
經驗法則 (1-bit, 0/π only):

rw 隨 aperture + width 增加：
  n=51 narrow cap:           rw=2  (R119)
  n=51 wide cap (20deg):     rw=3  (R129)
  n=71 narrow cap:           rw=5  (R133 NEW)
  n=71 wide cap:             rw=? (open, likely 7-8)

λ_mean 隨 optimization 難度降低：
  off-normal incidence:      λ=1   (R119 baseline)
  inc=0 + mmWave 28GHz:      λ=0.3 (R131)
  inc=0 + mmWave 38GHz:      λ=0.5 (R131)
  inc=0 + mmWave 60GHz:      λ=0.3 (R133 NEW)

Why 反向？
  - 寬 aperture / 寬帽蓋: ripple 容易爆 -> 強化 ripple penalty (rw上升)
  - inc=0 + 高頻 + 1-bit 量化: optimizer 自由度被 phase 量化壓縮 ->
    放鬆 mean penalty (λ下降) 讓 budget 給 worst-case
```

## Updated Patch Transition Boundary Map

| Configuration | n=51 | n=71 |
|---------------|------|------|
| Off-normal inc + narrow cap + sub-6G/mmWave ≤38GHz | R119 ✓ | (untested but likely OK) |
| inc=0° + 5.8GHz | R119 ✓ | (untested) |
| inc=0° + 28GHz | R131 (rw=2, λ=0.3) ✓ | (untested) |
| inc=0° + 38GHz | R131 (rw=2, λ=0.5) ✓ | (untested) |
| **inc=0° + 60GHz** | ❌ NO recipe | **R133 (rw=5, λ=0.3) ✓ NEW** |
| Wide cap broadside (20°) | R129 (rw=3, λ=1) ✓ | (untested) |
| Wide cap broadside (30°) | marginal (2/3) | (untested) |
| +45° steering | tied baseline | R127 needs re-tune |

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R132 | n=71 + R131 recipes -> 全 0/3 flat-top fail (recipe under-tuned) |
| R133 | n=71 grid 找到 sweet (rw=5, λ=0.3) -> PASS, boundary 破解 |

## 下一階段建議

1. **R134**: 把目前 boundary map 整合成一個 unified recipe selector function
   並在更多 (n, inc, freq) 點 cross-validate
2. **R135**: 1-bit + 量產 fab tolerance test (~1% phase noise) 驗證 robustness
3. **R136+**: surrogate-in-the-loop 架構準備（patch transition 階段）

## 結論

- R131 標的「inc=0°+60GHz physical boundary」**重新分類為 recipe limit**
- 真實的 1-bit + n=51 absolute boundary 還沒找到
- **可部署 1-bit recipe 涵蓋範圍擴大**：sub-6G + mmWave (28/38/60 GHz) 在 patch
  transition 都有對應 recipe 可用
- 但需要 (n, inc, freq, width) 4D recipe selector，不是單一 universal recipe
