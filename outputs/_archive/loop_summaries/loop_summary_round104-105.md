# /loop Round 104–105 兩輪總結 — Cross-Aperture Pareto + Compute Optimization

> R104 n=41 vs n=51 design space comparison。
> R105 smaller GD steps for compute budget reduction.
> Patch transition methodology 進入 budget-optimized 階段.

## TL;DR

| Round | Finding |
|-------|---------|
| R104 | n=51 + rw=2 是 reliable sweet spot; n=41 needs rw=5 for flat-top |
| R105 | 1000 steps -32% compute, 80% flat-top; 1500 仍是 production default |

## R104 — Cross-Aperture Design Space

| rw | n=41 best | n=51 best | flat-top n=41 / n=51 |
|----|-----------|-----------|----------------------|
| 0.0 | +6.52 | +7.35 | 0/5 / 0/5 |
| 0.5 | +3.69 | +6.69 | 0/5 / 0/5 |
| 1.0 | +3.91 | +5.44 | 0/5 / 2/5 |
| **2.0** | **+0.02** | **+1.92** | **2/5 / 5/5 ★** |
| 5.0 | -0.22 | -0.17 | 3/5 / 4/5 |

### Aperture sweet spot 不 transferable

- **n=51 + rw=2**: 100% flat-top reliable
- **n=41 + rw=2**: only 40% flat-top hit
- **n=41 needs rw=5** for reliable flat-top (60%)

→ Smaller patch (lower cost) → stricter ripple weight → lower max worst

### Patch design decision matrix

```
Cost-sensitive deployment:  n=41 + rw=5
Performance-critical:        n=51 + rw=2 (recommended)
```

## R105 — Smaller GD Steps for Compute

| Steps | Best worst | Flat-top | Time | -% Time |
|-------|-----------|----------|------|---------|
| 500 | +1.53 | 4/5 | 91s | -62% |
| 750 | **+2.32** | 4/5 | 126s | -47% |
| 1000 | +2.12 | 4/5 | 163s | -32% |
| **1500** | +1.92 | **5/5** | 238s | reference |

### Counter-intuitive

```
750 steps 給 highest best worst (+2.32) 但只 4/5 flat-top
1500 steps 給 100% flat-top reliability (+1.92)

Why? smaller steps:
  - 沒完全收斂 → 較多隨機性
  - 可能 lucky into different basin
  - 但 reliability 較低 (4/5 vs 5/5)

→ 1500 steps for production reliability (R99 confirmed)
  1000 steps for BO screening (-32% compute, accept 80% reliability)
```

### Patch budget optimization

```
Two-stage strategy:

Phase 1 (BO active learning, screening):
  GD steps = 1000
  -32% compute per evaluation
  Accept 80% flat-top hit rate
  Screen 100s of candidates fast

Phase 2 (Final deployment, verification):
  GD steps = 1500 (production default)
  100% flat-top reliability
  Run on top 10-20 from Phase 1

Total saving: ~25-30% across full deployment cycle
```

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R104 | Cross-aperture Pareto: n=51+rw=2 sweet spot; n=41 needs rw=5 |
| R105 | 1000 steps OK for screening, 1500 for final |

## 累計 (105 rounds, 142+ commits)

完整 budget-aware deployment guidance:

```
Compute budget reduction options:
  ✓ Use 1000 GD steps in BO screening (-32%)
  ✓ Multi-restart 5 seeds (already minimum)
  ✓ Stop early if surrogate confident (R86 cautious)

Aperture choice:
  ✓ n=51 + rw=2: production-grade, expensive
  ✓ n=41 + rw=5: cost-effective, lower max worst
  ✗ n>61: GPU cache thrash, infeasible (R97)
  ✗ n<31: limited DoF, hard for flat-top

Reliability targets:
  ✓ 100% flat-top hit (n=51 + rw=2 + 1500 steps): production
  △ 80% flat-top hit (n=51 + rw=2 + 1000 steps): screening
  △ 60% flat-top hit (n=41 + rw=5): cost-sensitive
```

## 對 Patch Antenna Final Recommendation Matrix (post-R105)

```
Patch design decision tree:

if cost_critical:
    use_arch = "small (n=41-equiv)"
    ripple_weight = 5.0
    expect: "lower max worst, ~60% flat-top hit"

elif performance_critical:
    use_arch = "large (n=51-equiv)"
    ripple_weight = 2.0
    expect: "max worst, 100% flat-top hit"

# Compute strategy:
phase1 = "BO screening, 1000 GD steps" (-32% compute)
phase2 = "Final verification, 1500 GD steps" (100% reliability)
```

## 結論

105 rounds RIS playground 完整探索 + 4-week deployment timeline 有 budget optimization +
aperture decision matrix + compute scheduling.

Patch transition 啟動 ready, 所有 deployment trade-offs 都 mapped.
