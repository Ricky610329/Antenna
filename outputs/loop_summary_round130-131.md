# /loop Round 130–131 兩輪總結 — 1-bit narrow-cap universal validation + inc=0 mmWave rescue

## TL;DR

| Round | Finding |
|-------|---------|
| R130 | 1-bit narrow-cap (10°) 跨 inc×freq 16 configs：**13/16 PASS**, all 3 fails at **inc=0° + mmWave** |
| R131 | 28/38 GHz 找到 rescue recipe (lower λ)，60 GHz 為 1-bit physical boundary |

**淨結果**：1-bit narrow-cap recipe 在多數 deployment 條件下 universal，但 (normal incidence × mmWave) 是已知 weak corner，需要 freq-aware recipe。

## R130 — 1-bit Narrow-cap Universal Test

R119 recipe (rw=2, λ=1, 1-bit) 在 width=10° broadside 跑 4 inc × 4 freq grid (n=51)：

### Worst-case heatmap

|  inc \\ freq | 5.8GHz | 28GHz | 38GHz | 60GHz |
|-----|--------|-------|-------|-------|
| 0°  | +3.12 | **+1.63** | **+0.54** | **+0.34** |
| 30° | +2.79 | +2.52 | +2.54 | +2.04 |
| 51° | +2.53 | +2.50 | +3.03 | +2.62 |
| 70° | +2.79 | +2.21 | +1.63 | +2.36 |

### Flat-top compliance

| inc \\ freq | 5.8GHz | 28GHz | 38GHz | 60GHz |
|-----|--------|-------|-------|-------|
| 0°  | OK | **2/5** | **1/5** | **1/5** |
| 30° | OK | 4/5 | OK | OK |
| 51° | OK | OK | 4/5 | 4/5 |
| 70° | 4/5 | OK | OK | OK |

### Pattern

- **inc=0° + mmWave** (28/38/60 GHz): 全部失敗
- **inc=0° + sub-6G** (5.8GHz): 通過
- **任意 off-normal incidence** (30/51/70°): 全部通過
- → **inc=0° (normal incidence) 是 1-bit + 38GHz+ 的 weak axis**

## R131 — inc=0° mmWave Rescue Grid

針對 R130 的 3 個 fail，重新 grid (rw, λ) at inc=0°, width=10°, 1-bit：

### inc=0°, 28GHz

```
rw=2 lam=0.3 → worst +2.85, side_mean -26.17, flat OK ★ BEST
rw=3 lam=0.3 → worst +0.54, OK
rw=3 lam=0.5 → worst +0.19, OK
```

### inc=0°, 38GHz

```
rw=2 lam=0.5 → worst +2.26, side_mean -22.84, flat OK ★ BEST
rw=5 lam=0.3 → worst -0.57, OK (negative worst, fail criterion)
```

### inc=0°, 60GHz

```
rw=2 lam=0.3 → worst +0.13, flat 2/3 (closest, but still fail)
ALL 12 (rw, lam) combinations: NO clean OK with positive worst
→ 1-bit + n=51 + inc=0° + 60GHz = PHYSICAL BOUNDARY
```

### Insight

```
Wide-cap problem (R129):  needs HIGHER rw (rw=2→3)
inc=0+mmWave problem:     needs LOWER lambda (1→0.3 or 0.5)

兩種 boundary 處理方式相反：
  - 寬帽蓋 ripple 太大 → 加強 ripple penalty
  - 高頻+normal入射 1-bit 量化太緊 → 放鬆 mean penalty
    讓 optimizer 專注 worst-case 約束
```

## Updated Width-aware × Inc-aware 1-bit Recipe Table

```python
def select_1bit_recipe(width_deg, inc_deg, freq_hz, n=51):
    """1-bit (0/pi only) RIS recipe selection, n=51."""
    
    # Narrow cap (<=10 deg) at inc=0deg + mmWave: special freq-aware
    if width_deg <= 15 and inc_deg == 0:
        if freq_hz <= 6e9:
            return {"rw": 2.0, "lam": 1.0}                          # R119 baseline
        elif freq_hz <= 30e9:
            return {"rw": 2.0, "lam": 0.3}                          # R131 28GHz rescue
        elif freq_hz <= 40e9:
            return {"rw": 2.0, "lam": 0.5}                          # R131 38GHz rescue
        else:  # >=60GHz
            raise ValueError("inc=0+60GHz exceeds 1-bit n=51 boundary")
    
    # Off-normal incidence (30-70 deg): R119 universal across freq
    if width_deg <= 15:
        return {"rw": 2.0, "lam": 1.0}                              # R119 baseline
    
    # Wide cap broadside (R129)
    if width_deg <= 20:
        return {"rw": 3.0, "lam": 1.0}
    if width_deg <= 30:
        return {"rw": 3.0, "lam": 0.5,
                "WARNING": "flat-top only 2/3, marginal"}
    
    raise ValueError(f"width {width_deg} deg exceeds 1-bit n=51 boundary")
```

## 累計 Validation 結果 (1-bit ONLY since R128)

| 軸 | Range tested | 狀態 | Recipe |
|---|---|---|---|
| Width × Steering | (10/20/30°) × (0/30/45°) | partial map | width-aware (R128/R129) |
| inc × freq @ width=10° broadside | 4 inc × 4 freq | 13/16 PASS | R119 baseline + R131 rescues |
| Normal incidence + mmWave | 28/38/60 GHz | 28/38 rescued, 60 boundary | freq-aware (R131) |
| Wide cap broadside | 20° / 30° | 20° rescued, 30° marginal | recipe-aware (R129) |

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R130 | 1-bit narrow-cap inc×freq grid，發現 inc=0+mmWave weak axis |
| R131 | 28/38 GHz 找到 rescue (lower λ)，60 GHz 為 1-bit boundary |

## 下一階段建議

1. **R132**: 1-bit + n=71 在 inc=0+60GHz 看 aperture upgrade 是否破 boundary
   （類似 R127 對 +45° steering 的 aperture rescue）
2. **R133**: 把所有 boundary 與 rescue recipe 整合成一個 deployment-ready
   recipe selector function 並驗證
3. **R134+**: surrogate-in-the-loop 架構準備（patch transition 階段）
