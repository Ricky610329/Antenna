# /loop Round 98–99 兩輪總結 — Final Closure + GD Steps Validation

> R98 寫 paper-style FINAL_REPORT.md (97 rounds consolidation)。
> R99 GD steps scaling validate 1500 default. Methodology fully closed.

## TL;DR

R98：`outputs/FINAL_REPORT.md` paper-style closure (8 sections)。
R99：1500 GD steps 是 deployment sweet spot；3000+ trade flat-top reliability for higher worst.

## R98 — Paper-style Closure

新增 `outputs/FINAL_REPORT.md`:
- Abstract (97 rounds 核心發現)
- Recommended pipeline code template
- Validated performance (Pareto, cross-freq, stress test)
- 13 cascade negative findings + remedies
- 4-week patch transition plan
- Visual + code deliverables

對 patch team: 這是 single-document reference for transition.

## R99 — GD Steps Scaling

| Steps | Best worst | Best ripple | Median | Flat-top hit |
|-------|-----------|-------------|--------|--------------|
| **1500** | +1.92 | 2.59 | +1.39 | **5/5 (100%) ✓** |
| 3000 | +3.01 | 2.55 | +1.78 | 4/5 |
| 5000 | +2.79 | 1.94 | +2.10 | 3/5 |

### Counter-intuitive Finding

更多 GD steps 提升 best worst (+1.09 dB at 3000)，**但 flat-top hit rate 下降** (5/5 → 3/5)。

### 為什麼 longer GD 反而 less reliable for flat-top

```
loss = -(main_min - side_max) + ripple_weight * (main_max - main_min)

深度收斂下:
  base loss saturate → optimizer 允許 main_max 上升換 main_min gain
  → ripple 增加, flat-top 違反

ripple_weight=2 是 fixed:
  early training: ripple penalty 顯著 → flat-top 收斂
  late training: 相對失效 → 偏向 sharp peak

→ 1500 steps 是 deployment sweet spot
  3000+ 適合 research max records (不在意 flat-top)
```

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R98 | FINAL_REPORT.md paper-style consolidation |
| **R99** | **1500 GD steps validated as deploy sweet spot** |

## 累計 (99 rounds, 134+ commits) — Hyperparameter Validation Matrix

| Hyperparameter | Validated | Reference round |
|---------------|-----------|-----------------|
| Loss design | worst-case + ripple penalty | R64 |
| Ripple weight | rw=2 (flat-top sweet spot) | R65, R94 |
| Aperture n | 41-51 (sweet spot, n>61 cache limit) | R51, R92, R97 |
| Beta soft-max | β=20 (close to true min/max) | R64 |
| Frequency | 28/38/60 GHz all work | R96 |
| Multi-restart | 5 seeds adequate | R44, R89 |
| **GD steps** | **1500 (deployment), 3000 (research)** | **R99** |
| Architecture | CNN het ensemble c={16,32,64} d={3,4,5} | R68, R89 |
| Dataset balance | 1:1 random/optimized | R83-R84 |
| BO acquisition | UCB κ=2.0 | R89 |
| Surrogate use | ranking only, NOT GD | R90 |

完整 patch transition reference 已 codified。
