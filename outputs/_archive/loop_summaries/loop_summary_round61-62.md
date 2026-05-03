# /loop Round 61–62 兩輪總結 — Aperture Scaling 主導

> 接續 R59-60 的 38 GHz × 15 / × 21 紀錄。R61-62 持續向更大 aperture 推進，
> 達 **+27.45 dB**（38 GHz × n=31）。

## TL;DR — 雙輪三破紀錄

| Round | 紀錄 | 配置 |
|-------|------|------|
| R60 | +23.88 | 38 GHz × n=21 |
| R61 | +24.97 | 38 GHz × n=25 |
| **R62** | **+27.45 ★** | **38 GHz × n=31** |

從 v1 (-4.08) 到 R62 (+27.45) = **+31.53 dB 累計改善**

## Round 61 — 38 GHz × n=23/25/27 + 28 GHz × n=15-21

### 38 GHz n sweep（接續 R60）

| n | aperture | best | mean |
|---|----------|------|------|
| 21 (R60) | 10.5λ | +23.88 | +22.55 |
| 23 | 11.5λ | +22.16 | +21.50 |
| **25** | **12.5λ** | **+24.97 ★** | +22.69 |
| 27 | 13.5λ | +24.21 | +22.36 |

### 28 GHz n sweep（驗證 universality）

| n | best | mean |
|---|------|------|
| 13 (R57) | +21.31 | +15.89 |
| 15 | +19.07 | +16.62 |
| 17 | +21.31 | +18.06 |
| 19 | +20.28 | +19.46 |
| **21** | **+23.45 ★** | **+21.88** |

→ 28 GHz × n=21 達 **+23.45**，破 R57 +21.31 by +2.14。

## Round 62 — 38 GHz × n=29/31 + n=25 多 seed

### n=29/31 × 5 seeds

| n | aperture | best | mean |
|---|----------|------|------|
| **29** | **14.5λ** | **+27.01 (seed 1)** | **+24.95** |
| **31** | **15.5λ** | **+27.45 (seed 1) ★** | **+25.77** |

### n=25 multi-seed (seeds 5-14)

新增 seed=13 達 **+25.03**，超過 R61 best +24.97。

15 seeds 統計：
- max = +25.03 (seed 13)
- mean ≈ +23.0
- 所有 seeds ≥+20.76 → **robust**

## Aperture Scaling 規律（free-phase 完整版）

```
n   | aperture | best  | trend
----+----------+-------+------
11  | 5.5λ     | +15.51|  ↑
13  | 6.5λ     | +18.12|  ↑
15  | 7.5λ     | +23.02|  ↑↑
17  | 8.5λ     | +20.65|  ↓
19  | 9.5λ     | +21.69|  ↑
21  | 10.5λ    | +23.88|  ↑
23  | 11.5λ    | +22.16|  ↓
25  | 12.5λ    | +25.03|  ↑↑
27  | 13.5λ    | +24.21|  ↓
29  | 14.5λ    | +27.01|  ↑↑
31  | 15.5λ    | +27.45|  ↑ NEW RECORD
```

整體 monotonic 向上 + 局部 fluctuation。**沒有 sweet aperture，是
"越大越好" + 個別 seed lucky**。

物理理解：
- 元素數 N → 理論 array gain 上限 10·log10(N)
- N=31² = 961 → +29.83 dB
- 我們 binary 達 +27.45 dB → ~93% of theoretical
- 留下 ~2.4 dB 是 1-bit quantization loss（實證接近文獻 3 dB 理論）

## Sigmoid vs Free-Phase 對 Aperture 偏好的差異

| Path | 28 GHz best n | 38 GHz best n |
|------|---------------|---------------|
| Sigmoid (R51/R53) | 13 (6.5λ) | 15 (7.5λ) |
| **Free-phase (R57-R62)** | **21+ (10.5λ+)** | **31+ (15.5λ+)** |

差異原因：
- Sigmoid 半圓 phase aliasing 隨 N 變嚴重 → 卡 local optimum
- Free-phase 全圓無 aliasing → larger N 給更多 DoF, 更接近理論上限

## 累計（62 rounds, 97+ commits）

### 紀錄歷程
```
v1                                          −4.08 dB
v6 generator best                           −0.46 dB
GD+SA reheat=2 (R25)                        +9.75 dB
SA-per-restart 28 GHz × 13 (R48)           +13.44 dB
Free-phase 28 GHz × 13 (R57)               +21.31 dB
Free-phase 38 GHz × 15 (R59)               +23.02 dB
Free-phase 38 GHz × 21 (R60)               +23.88 dB
Free-phase 38 GHz × 25 (R61)               +24.97 dB
Free-phase 38 GHz × 31 (R62)               +27.45 dB ★ NEW GLOBAL RECORD
```

從 v1 (-4.08) 到 R62 (+27.45) = **+31.53 dB 累計改善**

## Open Questions（更新）

1. **n=35/41 at 38 GHz** 是否再突破？(R63 探)
2. 28 GHz × n=29/31 是否也達 +25+？(R63 探)
3. 是否有 1-bit theoretical ceiling？(目前 ~93% array gain)
4. 5.6 GHz / 60 GHz × 大 n free-phase performance？
5. inc/width fine grid 在 free-phase 下的 sweet spot
