# /loop Round 128–129 兩輪總結 — 課程修正：1-bit ONLY + Width-aware Recipe

## 課程修正

使用者新增關鍵 constraint：「**整能用 0 or 180 度的相位去做 ris pattern**」
→ 實際 deployable hardware 是 1-bit 二進位 phase (0 或 π only)。
→ R121 CHAMPION (2-bit + λ=1) 不符合 deployment constraint。
→ Pivot 回 R119 (1-bit + λ=1 + rw=2)。

並新增兩個探索維度：
- 不同寬度的帽蓋 (cap width)
- 不同角度 (steering angle)

## R128 — 1-bit Width × Steering Joint Sweep

R119 recipe (1-bit, rw=2, λ=1) 跨 width × steering grid (n=51, 38GHz, inc=51°)：

| width\steer | 0° | +30° | +45° |
|-------------|-----|------|------|
| 10° | +3.03, **4/5** | +1.41, 4/5 | +1.28, ✓ |
| 20° | +3.02, **3/5** | +1.07, ✓ | +0.69, ✓ |
| 30° | +3.39, **2/5** | +0.99, 3/5 | +0.66, 4/5 |

### Findings
1. **Narrow cap (10°)**: 跨 steering 全部 robust，R119 recipe 直接可用
2. **Wide cap broadside**: flat-top 隨 width 退化（4/5 → 3/5 → 2/5）→ recipe 強度不夠
3. **Wide cap + extreme steering**: worst-case 縮但 flat-top 反而 OK
   （steering 自然限制 worst-case 大小，反而沒 over-shoot）
4. **side_mean** 整片 -22 ~ -29 dB 跨所有 configs，recipe core 仍 functional

## R129 — Wide-cap (rw, λ) Re-grid

針對 R128 發現的 wide-cap broadside flat-top 問題，重新 grid (rw, λ)：

### Width = 20° broadside

| rw | λ=0.5 | λ=1.0 | λ=1.5 |
|----|-------|-------|-------|
| 2  | +2.06, 2/3 | +3.02, 2/3 (R119) | +3.56, 0/3 |
| **3** | **+0.69, ✓** | **+1.20, ✓** ★ | +0.89, 0/3 |
| 5  | +0.25, ✓ | +1.31, 1/3 | +0.25, 2/3 |
| 8  | -0.27, ✓ | -0.31, ✓ | -0.52, 2/3 |

**Sweet spot**: **rw=3, λ=1.0** → worst +1.20, side_mean -22.23, flat-top ✓

### Width = 30° broadside

| rw | λ=0.5 | λ=1.0 | λ=1.5 |
|----|-------|-------|-------|
| 2 | +2.16, 1/3 | +3.39, 1/3 | +2.45, 0/3 |
| 3 | +1.11, 2/3 | +1.03, 1/3 | +0.81, 1/3 |
| 5 | -0.10, 0/3 | +0.68, 1/3 | +1.25, 1/3 |
| 8 | -1.32, 0/3 | -0.45, 2/3 | -1.11, 0/3 |

**No clean OK recipe.** Best partial: rw=3 λ=0.5 (2/3 flat-top, worst +1.11).
→ Width=30° 是 **1-bit + n=51 的 width-axis physical boundary**

## Width-aware 1-bit Recipe Table

```python
def select_1bit_recipe(width_deg, n=51, freq=38e9):
    """Select recipe based on cap width."""
    if width_deg <= 15:
        return {"rw": 2.0, "lam": 1.0}  # R119 baseline
    elif width_deg <= 20:
        return {"rw": 3.0, "lam": 1.0}  # R129 sweet
    elif width_deg <= 30:
        return {"rw": 3.0, "lam": 0.5,
                "WARNING": "flat-top only 2/3, consider larger n or narrower spec"}
    else:
        raise ValueError("width > 30deg exceeds 1-bit + n=51 physical boundary")
```

## 與 R121 (2-bit) 對比 — Trade-off 表

| Hardware | width=15° broadside | side_mean | Universal? |
|----------|---------------------|-----------|------------|
| 1-bit (R119/R128) | worst +3.03 | -29.11 | only narrow cap |
| 1-bit wide (R129) | worst +1.20 (w=20°) | -22.23 | recipe shrinks worst |
| 2-bit (R121, NOT deployable) | worst +3.45 | -30.84 | universal across 4 axes |

**1-bit cost**: ~1-3 dB worst-case + width-specific recipe needed。
但這是 deployment-realistic recipe。

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R128 | 1-bit width × steering grid, 找到 recipe boundary |
| R129 | Width-aware re-grid, width=20° 找到 sweet spot, width=30° boundary |

## 下一階段建議

1. **R130**: width=10° + 不同 inc/freq 確認 narrow cap 是 1-bit universal sweet
2. **R131**: 更系統 width sweep (5° / 8° / 12° / 15° / 18° / 22° / 25°) 找精確 boundary
3. **R132+**: 1-bit + n=71 看 aperture 是否能突破 width=30° boundary
   （類似 R127 對 +45° steering 的 aperture rescue）
