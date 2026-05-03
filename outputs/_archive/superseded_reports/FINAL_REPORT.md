# Binary RIS Pattern Optimization — 97 Rounds Final Report

> Paper-style consolidation of 97 rounds RIS playground exploration.
> 給 patch antenna team 的 transition reference + complete experimental record.

---

## Abstract

This report consolidates 97 rounds (~3 weeks) of systematic exploration on
binary 1-bit Reconfigurable Intelligent Surface (RIS) pattern optimization,
with the explicit goal of **establishing a trustworthy methodology for
patch antenna design via surrogate-in-the-loop optimization**.

Key finding: **R57-R63 single-headline-record optimization (max-max metric, +30 dB)
is virtual** — the same patterns achieve only -18 dB worst-case suppression on
the same physical hardware. **Worst-case loss with ripple penalty (R64-R94)
gives genuinely deployable solutions** (n=51, +1.92 dB worst, ripple 2.6 dB,
100% flat-top reproducibility).

**Surrogate-in-the-loop is partially viable** for patch:
- ✓ BO acquisition via heterogeneous CNN ensemble + UCB (R89)
- ✗ GD-through-surrogate fails (R77/R79/R80/R90: gradient quality random)

Recommended patch transition: 3-4 weeks via active learning on 200-1500
balanced HFSS entries.

---

## 1. The Problem

Phase-quantized RIS patterns (1-bit, {0, π}) are difficult to optimize:
- 2^(n²) discrete configurations (intractable)
- Continuous relaxation introduces quantization gap
- (target_spec → pattern) mapping is multimodal (hamming 51.72%, R71)

Original "headline" metric (max-max) rewards **one-point sharp peaks**, not
broad deployable patterns. Patch antenna methodology inheriting this metric
will produce solutions that fail HFSS verification.

---

## 2. Recommended Pipeline (R64-R94)

```python
def deploy_one_target(spec, n_restarts=5, gd_steps=1500, ripple_weight=2.0):
    sim = RISSimulator_or_HFSS(spec.geometry)
    main_lo, main_hi = build_main_idx(spec.target_theta_c, spec.target_width)
    
    best = None
    for seed in range(n_restarts):  # multi-restart, R44/R56
        torch.manual_seed(seed)
        # Free-phase parameterization (R57)
        params = nn.Parameter(torch.rand(n, n) * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)
            # Worst-case + ripple penalty (R64)
            loss = worst_case_loss(resp, main_lo, main_hi, 
                                   beta=20.0, ripple_weight=ripple_weight)
            loss.backward()
            opt.step()
        
        # Optimal 1-bit quantization (R57)
        phase = (params * π) % (2π)
        binary = ((phase > π/2) & (phase < 3π/2)).float()
        
        if eval(binary) > eval(best):
            best = binary
    
    return best


def worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=2.0):
    main = resp[..., main_lo:main_hi]
    side = torch.cat([resp[..., :main_lo], resp[..., main_hi:]], dim=-1)
    
    # logsumexp soft-min/max (smooth, β=20 close to true min/max)
    main_min = -(1/beta) * logsumexp(-beta * main)
    side_max =  (1/beta) * logsumexp( beta * side)
    
    loss = -(main_min - side_max)  # maximize worst-case suppression
    if ripple_weight > 0:
        main_max = (1/beta) * logsumexp(beta * main)
        loss = loss + ripple_weight * (main_max - main_min)  # constrain ripple
    return loss
```

---

## 3. Validated Performance (Reference Benchmarks)

### Pareto frontier at n=51 (R94, 38 GHz, 15° broadside flat-top)

| ripple_weight | best worst | best ripple | flat-top hit | use case |
|---------------|-----------|-------------|--------------|----------|
| 0.0 | +7.35 | 10.98 | 0/5 | high-gain steering |
| 0.5 | +6.69 | 6.72 | 0/5 | mostly steering |
| 1.0 | +5.44 | 5.31 | 2/5 | mixed |
| **2.0** | **+1.92** | **2.59** | **5/5** ★ | flat-top deployment |
| 5.0 | -0.17 | 2.02 | 4/5 | extreme flatness |

### Cross-frequency (R96)

| Freq | Best worst | Flat-top hit |
|------|-----------|--------------|
| 28 GHz | +1.66 | 2/5 |
| 38 GHz | +1.92 | **5/5** |
| 60 GHz | +2.09 | 3/5 |

### Stress test (R95): graceful degradation

| Spec difficulty | Worst | Flat-top |
|-----------------|-------|----------|
| Easy (broadside) | +1.92 | 100% |
| Off-axis (-25°) | +0.63 | 20% |
| Wide (25°) | +0.91 | 0% |
| Combined hard | -0.57 | 0% |

---

## 4. Cascade Negative Findings (Avoid These Traps)

### Loss / Metric (R57-R85)

```
✗ max-max metric:                R63 +30 dB virtual, real -18 dB
✓ worst-case + ripple:           R94 deployable

✗ scalar metric surrogate:       R69 mean collapse
✓ full S-curve surrogate:        R68 dense supervision

✗ supervised BCE on geometry:    R73 garbage averages
✓ end-to-end via differentiable: R74 continuous works
✗ binary STE training:           R75 fail (architectural)
```

### Surrogate (R77-R90)

```
✗ trust function MAE alone:      R77 adversarial gap 13 dB
✗ Sobolev fix gradient:          R80 architectural limit
✗ ranking on optimized-only:     R81 Spearman 0.03
✓ diverse balanced dataset:      R82-R84 Spearman 0.6
✗ 1:6 imbalance scaling:         R83 hurts
✓ 1:1 balanced:                  R84 best ranking
✗ mode-specific surrogate:       R87 95% Spearman drop
✓ mixed-mode + conditioning:     R87 contrastive learning
```

### Active Learning (R85-R89)

```
✗ greedy acquisition:            R85 worse than random
✗ same-arch ensemble:            R86 std too small
△ MC Dropout:                    R88 tied random
✓ heterogeneous ensemble UCB:    R89 first to beat random
```

### Deployment (R90, R97)

```
✗ GD-through-surrogate:          R77/R79/R80/R90 cos 0
✓ surrogate for ranking only:    R89 BO works
✓ HFSS-direct for optimization:  R90 dichotomy

Computing budget (RIS sim 38 GHz):
  n=41: 30s/restart, 2.5 min/target (sweet spot)
  n=51: 6 min/restart, 30 min/target (cache bottleneck)
  Patch HFSS ≈ RIS n=51 wall-clock
```

---

## 5. Patch Antenna Transition (Concrete Plan)

### Phase 1: Initial dataset (Week 1-2)

```
HFSS 200 entries:
  - 100 random geometries
  - 100 GD-optimized geometries (basic worst-case search)
  - Mixed-mode (multi use case targets) ← R87
  - Class balance 1:1 (random:optimized) ← R83-R84

Saved per entry:
  - geometry parameters
  - full S-curve (or radiation pattern)
  - Worst-case + ripple metrics
  - Multi-seed log
```

### Phase 2: Surrogate training (Week 2-3)

```
Heterogeneous ensemble (R89 winning recipe):
  - CNN(c=16, d=3) - small
  - CNN(c=32, d=4) - medium
  - CNN(c=64, d=5) - large
  + dropout=0.3 for additional MC option

Architecture: full S-curve output (R69 dense supervision)

4-tier validation (R77-R81):
  ✓ Function MAE < 1 dB
  ✓ Spearman ranking > 0.5 (BO threshold)
  △ Gradient cosine > 0.7 (likely fail, OK)
  ✓ Adversarial gap < 5 dB
```

### Phase 3: Active Learning BO (Week 3-4)

```
UCB acquisition:
  preds = stack([m(candidates) for m in ensemble])
  mean = preds.mean(0)
  std = preds.std(0)  # ensemble variance
  ucb = mean + 2.0 * std  # κ=2.0 (R89)
  selected = candidates[ucb.argsort()[-K:]]

Run HFSS on selected, add to dataset, retrain.

Maintain class balance during expansion (R83).
```

### Phase 4: Final deployment (Week 4)

```
Primary: HFSS-direct optimization with worst-case loss
  - 5 restarts per target
  - Per-target ~25 min (cluster batched)

Acceleration: surrogate for screening (NOT for GD)
  - Pre-filter 100 candidates via surrogate
  - Run HFSS on top-K selected by UCB
```

---

## 6. Visual Deliverables

| Image | Content |
|-------|---------|
| `outputs/r93_max_max_vs_worst_case.png` | R63 (max-max 虛胖 -18 dB) vs R92 (worst-case +1.92 dB) side-by-side |
| `outputs/r94_pareto_n51.png` | Deployment design space map (worst vs ripple vs ripple_weight) |
| `outputs/r91_deployment_demos/flat_top_38GHz.png` | Concrete deployable example (main beam 整片貼上蓋) |
| `outputs/best_record_38ghz_n41.png` | Original R63 record (visually shows sharp peak issue) |
| `outputs/aperture_scaling.png` | Aperture vs suppression (R57-R63 max-max regime) |

---

## 7. Code Deliverables

```
script/
├── PATCH_METHODOLOGY.md           ← Transition reference (13 sections)
├── methodology_demo.py            ← Recommended pipeline (R91)
├── verify_free_phase_record.py    ← Free-phase + worst-case (R57-R64)
├── optimize_worst_case.py         ← Per-target GD (R64)
├── train_surrogate.py             ← Forward CNN surrogate (R68)
├── train_metric_surrogate.py      ← Failed: scalar metric (R69)
├── train_conditional_generator.py ← Failed: supervised BCE (R73)
├── train_e2e_generator.py         ← Failed: STE training (R75)
├── train_sobolev_surrogate.py     ← Failed: gradient supervision (R80)
├── active_learning_demo.py        ← Failed: greedy (R85)
├── active_learning_ucb.py         ← Failed: same-arch ensemble (R86)
├── active_learning_mc_dropout.py  ← Marginal: MC Dropout (R88)
├── active_learning_het_ensemble.py← ✓ Het ensemble UCB (R89)
├── measure_gradient_quality.py    ← Diagnose R79
├── het_ensemble_gradient_quality.py← Confirm dichotomy R90
├── surrogate_ranking_quality.py   ← Pareto rank test R81
├── compare_surrogates.py          ← R72 v1 vs v2
├── build_dataset.py               ← Pareto frontier dataset (R66)
├── build_dataset_v3_diverse.py    ← Diversity (R82)
├── filter_dataset_by_rw.py        ← Mode filtering (R87)
└── (15+ more)

outputs/
├── dataset_v1/, v2/, v3/, v4/, v5/  ← Progressive improvements
├── r91_deployment_demos/             ← End-to-end examples
├── loop_summary_round*.md            ← 47 detailed records
└── (visualizations + .npy data)
```

---

## 8. Conclusion

**RIS binary 1-bit playground is exhausted as a methodology validation environment.**
All 7 positive design rules + 13 cascade negative findings + winning BO recipe
are codified.

**Patch antenna team is ready to begin transition** with:
- Concrete pipeline (R91 demo recipe)
- Realistic compute budget (R97 timing, 3-4 weeks total)
- Validated robustness (R95 stress, R96 cross-freq)
- Deployment design space map (R94 Pareto)
- Code templates (R76 PATCH_METHODOLOGY.md)

The core message for patch deployment:

> **Don't chase headline numbers. Optimize for worst-case + ripple.
> Use surrogate for ranking, HFSS for optimization. Maintain dataset
> diversity + class balance. Validate at 4 tiers before deploy.**

---

*Final report compiled across rounds R1-R97 (2026-04-29 to 2026-05-01).*
*Ready for patch antenna methodology transition.*
