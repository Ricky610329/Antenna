# /loop Round 92–93 兩輪總結 — Aperture Scaling Confirmed + R63 vs R92 視覺對比

> R92 確認 aperture scaling 規律延伸到 worst-case constrained regime。
> R93 視覺化對比 R63 (max-max 虛胖) vs R92 (worst-case deployable),
> 給 patch team concrete reference 看「整片貼上蓋」vs「單一尖峰騙 metric」的差異。

## TL;DR

| Method | n | best result | flat-top |
|--------|---|-------------|----------|
| R63 max-max | 41 | headline +30.55, **worst -18.21** (虛胖) | **75/80 違反帽蓋** ✗ |
| R91 worst-case | 41 | worst +0.26, ripple 1.36 | ✓ |
| **R92 worst-case** | **51** | **worst +1.92, ripple 2.59** | **0/30 ✓ ★** |

**R92 n=51 是真實 deployable single-config record**.
新紀錄 from R91 的 +0.26 → R92 的 +1.92 (+1.66 dB improvement)。

## R92 — Aperture Scaling 在 flat-top regime 仍成立

### 設計

R57-R63 max-max 路線發現「aperture 越大越好」。
R92 test: 在 R64 worst-case loss + flat-top constraint 下是否一樣?

Pipeline (R76-R85 final recommendation):
- Free-phase parameterization (R57)
- Worst-case loss + ripple penalty rw=2 (R64)
- Multi-restart 10 seeds
- Optimal 1-bit quantization

### 結果

| n | Aperture | Elements | best worst | ripple | flat-top |
|---|----------|----------|-----------|--------|----------|
| 41 (R91) | 20.5λ | 1681 | +0.26 dB | 1.36 | ✓ |
| **51 (R92)** | **25.5λ** | **2601** | **+1.92 dB** | **2.59** | **✓** |
| 61 | 30.5λ | 3721 | (interrupted, computation slow) | — | — |

n=51 比 n=41:
- worst suppression +1.66 dB
- ripple slightly larger (2.59 vs 1.36, both still flat-top)
- 兩個都通過 flat-top compliance

### 物理解釋

```
Aperture scaling 在 max-max 與 worst-case 兩 regime 都成立:
  More elements → more phase DoF
  → Better optimization solution space
  → Higher achievable worst-case suppression

But trade-off:
  Larger pattern = more compute (n=61 太慢)
  More elements = more hardware cost
  Sweet spot 取決於 application
```

## R93 — 視覺對比 (R63 max-max 虛胖 vs R92 worst-case deployable)

### Side-by-side 視覺證明

`outputs/r93_max_max_vs_worst_case.png` shows:

**R63 (top, max-max optimization)**:
- Binary pattern 41×41 (quasi-random looking)
- Response curve: 中心一根 sharp peak, main beam region (40° wide, green) 大部分在 -3 dB 以下
- Distribution: 75/80 main 點 < -3 dB (94% 違反帽蓋)
- Headline +30.55 dB BUT worst -18.21 dB
- "Optimization works on paper, fails on deployment"

**R92 (bottom, worst-case + ripple penalty)**:
- Binary pattern 51×51
- Response curve: main beam 真的是 flat plateau, 整片在 -3 dB 以上
- Distribution: 0/30 main 點 < -3 dB ★
- Worst +1.92 dB (lower number, but真實可部署)
- "Lower headline metric, but actually deployable"

### Pivotal Lesson

**Single absolute number doesn't tell the story**:
- max-max metric +30.55 sounds impressive
- worst-case +1.92 sounds modest
- 但 R92 才是真實能用 (整片貼上蓋), R63 不能 (尖峰騙 metric)

對 patch team 的核心 message:
- Don't chase max-max records
- Look at full distribution: main beam shape, sidelobe envelope
- Worst-case + ripple metrics 反映真實 deployment performance

## 紀錄歷程更新

| Round | Best | 評估方式 | 真實 deployable? |
|-------|------|---------|-----------------|
| R57-R63 | +30.99 | max-max | ✗ (R64 重評實為 -18) |
| R64 | +6.88 | worst-case (width=30) | △ (n=41, ripple high) |
| R91 demo | +0.26 | worst + flat-top | ✓ |
| **R92** | **+1.92** | **worst + flat-top, n=51** | **✓ NEW DEPLOYABLE BEST** |

## 累計（93 rounds, 127+ commits）

完整 patch transition reference deliverables:
- `script/PATCH_METHODOLOGY.md`: 13-section reference (R76-R89 lessons)
- 44 round summaries
- 5 datasets (v1-v5)
- 6+ surrogate variants
- 4 deployment demo + 1 visual side-by-side
- Cascade negative findings + remedies (R77-R90)
- 1 winning BO recipe (R89 het ensemble UCB)

## 對 Patch Team 最終 Action Items（FINAL FINAL）

```
Phase 1 (Week 1-2): Initial dataset
  ✓ HFSS 200 entries: 100 random + 100 GD-optimized
  ✓ Mixed-mode (R87)
  ✓ Class balance 1:1 (R83-R84)
  ✓ Worst-case + ripple penalty labels (R64)

Phase 2 (Week 2-3): Surrogate
  ✓ Heterogeneous ensemble c={16,32,64} d={3,4,5} (R89)
  ✓ Dropout 0.3 for MC option (R88)
  ✓ 4-tier validation (R77-R81)

Phase 3 (Week 3+): BO + Deployment
  ✓ UCB κ=2.0 acquisition (R89)
  ✓ HFSS-direct optimization (R90)
  ✓ Reference benchmarks (R91-R92):
    - Steering rw=0: high suppression but no flat-top
    - Flat-top rw=2: lower number but真實 deployable
    - Aperture larger → better deployable (R92)
  ✗ NEVER GD-through-surrogate
  ✗ NEVER greedy AL
  ✗ NEVER mode-specific surrogate
  ✗ NEVER trust max-max metric

Performance expectations (RIS extrapolated to patch):
  Patch 連續 geometry → smoother optimization
  預期 < 30 geom params 達 RIS n=51 級 (worst +2 dB, flat-top)
  Manufacturing cost lower than RIS (no per-pixel control)
```

## 結論

**93 rounds RIS playground 完整 saturate**。

最重要 deliverable:
- R63 vs R92 視覺對比 (`outputs/r93_max_max_vs_worst_case.png`)
- 直接展示 "max-max 虛胖" vs "worst-case 真實 deployable" 的差異
- Patch team 看一次就理解 methodology 核心 insight

下一階段建議: 啟動 patch antenna 真實資料收集 + apply this methodology。
