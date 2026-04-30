# /loop Round 90–91 兩輪總結 — Final Dichotomy + End-to-End Demo

> R89 het ensemble UCB 贏 random (BO recipe)。R90 確認 het ensemble 不修
> gradient (deployment dichotomy 確立)。R91 跑完整 recommended pipeline 在三個
> concrete specs 上，給 patch team reference deliverable。

## TL;DR

| Round | 結果 |
|-------|------|
| R90 | Het ensemble gradient cos -0.006 (still random). 確認 dichotomy: surrogate for BO ✓ / surrogate for GD ✗ |
| R91 | 三個 deployment specs 跑完整 pipeline, flat_top 達 worst +0.26 dB, ripple 1.36 dB, 整片貼上蓋 ✓ |

## R90 — Het Ensemble Gradient (Dichotomy 確認)

### 設計

R89 het ensemble (channels 16/32/64, depths 3/4/5) 解決 BO ranking。
Test 是否也解決 R79 gradient quality issue (cos 0.001 random)。

### 結果

| Method | Gradient cos similarity |
|--------|-------------------------|
| R79 single CNN | +0.001 |
| **R90 het mean prediction** | **-0.006** |
| **R90 het avg gradients** | **-0.007** |
| Pass threshold | > 0.7 |

**Het ensemble fix R89 ranking BUT 不修 gradient**。

### 為什麼

```
Ranking 是 collective property:
  Multi-model ensemble mean prediction 較準
  Errors cancel out
  Het architectures give complementary biases
  → Improvement

Gradient 是 pointwise property:
  每個 model 的 ∂y/∂x 是 random
  Average of random = random
  No mechanism for ensemble to correct gradient
  → No improvement
```

### Final Patch Deployment Dichotomy

```
✓ Surrogate ensemble for:
  - BO ranking (R89: het ensemble +5.19 vs random +4.79)
  - Design space screening
  - Coarse evaluation

✗ Surrogate ensemble NEVER for:
  - GD-through-surrogate (R77/R79/R80/R90 all fail)
  - High-precision gradient-based optimization

✓ HFSS-direct for:
  - Per-target final optimization (worst-case loss + multi-restart)
  - Verification of surrogate-selected candidates
```

## R91 — End-to-End Methodology Demo

### 推薦 Pipeline

```python
def recommended_deployment(spec):
    sim = HFSS_or_RIS_sim(spec.geometry_params)
    best = None
    for seed in range(10):  # multi-restart (R44/R56)
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(...) * 2.0)  # free-phase init (R57)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(1500):  # GD steps
            resp = sim(params)
            loss = worst_case_loss(  # R64 worst-case + ripple
                resp, spec.main_lo, spec.main_hi,
                ripple_weight=spec.ripple_weight,
            )
            loss.backward(); opt.step()
        # Optimal 1-bit quantization (R57)
        binary = quantize_optimal(params)
        if eval(binary) > eval(best): best = binary
    return best
```

### 三個 deployment specs 實測

#### Spec 1: Narrow Steering (rw=0, 38 GHz × n=31, 7° wide)
```
Best worst:    +6.19 dB
Median:        +3.44 dB
Mean ± std:    +3.76 ± 1.44
Flat-top hit:  0/10 (預期: rw=0 no ripple penalty)
Ripple avg:    14 dB
Use case:      點對點高增益 link
```

#### Spec 2: Flat-Top (rw=2, 38 GHz × n=41, 15° wide)
```
Best worst:    +0.26 dB
Median:        -0.81 dB
Mean ± std:    -1.03 ± 1.00
Flat-top hit:  5/10 ★ (50% achievement)
Ripple avg:    ~3 dB
Use case:      廣域覆蓋, multi-user
Best seed image: outputs/r91_deployment_demos/flat_top_38GHz.png
  - Main beam 真的是 flat plateau (not peak)
  - ripple 1.36 dB
  - sidelobe cluster -15~-25 dB
  - main beam clustered above -3 dB cap
```

#### Spec 3: Off-Axis (rw=1, 28 GHz × n=31, 10° wide @ θ_c=-30°)
```
Best worst:    +0.30 dB
Median:        -0.87 dB
Mean ± std:    -1.08 ± 0.85
Flat-top hit:  7/10 ★ (70% achievement)
Ripple avg:    ~2.7 dB
Use case:      Off-axis 不顯著 harder than broadside
```

### Key Takeaways for Patch

```
1. Multi-restart 10 seeds:
   - 50-70% flat-top achievement rate (5-7 lucky seeds)
   - Best seed worst_supp 比 median 高 ~1 dB
   
2. Use case → ripple weight 對應:
   rw=0: max suppression, no flat-top
   rw=1: balanced (~2.7 dB ripple)
   rw=2: strict flat-top, lower suppression
   
3. Off-axis vs broadside: similar performance
   Pattern 設計有彈性, 不必固定 broadside

4. 預期 patch performance (extrapolation):
   - 連續 geometry → smoother loss landscape
   - 預期 50-80% flat-top achievement (vs RIS 50-70%)
   - 預期 worst_supp 範圍類似 (single mode constrained)
```

## 紀錄歷程更新（最終 90+ 對照表）

| Round | 結果 |
|-------|------|
| R57-R63 | +30.99 dB (max-max, 虛胖) |
| R64-R65 | +6.88 dB worst-case |
| R66-R72 | dataset/surrogate scaling |
| R71 | (config → pattern) multimodal hamming 51.72% |
| R76 | methodology distill |
| R77-R88 | cascade 11 negative findings |
| R89 | **het ensemble UCB beats random ★** |
| R90 | gradient dichotomy 確認 |
| **R91** | **end-to-end demo: flat-top 0.26 worst, 1.36 ripple, 整片貼上蓋** |

## 累計（91 rounds, 125+ commits）

完整 deliverables:
- `script/PATCH_METHODOLOGY.md`: 13-section reference
- 43 round summaries
- 5 dataset versions (v1-v5)
- 6+ surrogate variants
- 5 generators / active learning approaches
- 3 deployment demo specs (R91 reference)

## 對 Patch Team 最終 Action Items (post-R91, FINAL)

```
Phase 1 (Week 1-2): Initial dataset
  ✓ HFSS 200 entries: 100 random + 100 GD-optimized
  ✓ Mixed-mode (R87)
  ✓ Class balance 1:1 (R83-R84)
  ✓ Worst-case loss labels (R64)

Phase 2 (Week 2-3): Surrogate
  ✓ Het ensemble 3 archs (R89): c={16,32,64} d={3,4,5}
  ✓ + dropout 0.3 for additional MC option (R88)
  ✓ 4-tier validation (R77-R81)

Phase 3 (Week 3+): BO + Deployment
  ✓ UCB κ=2.0 acquisition (R89)
  ✓ HFSS-direct for actual optimization (R90)
  ✓ 用 R91 demo as reference for expected performance
  ✗ NEVER GD-through-surrogate (R77/R79/R80/R90)
  ✗ NEVER greedy AL (R85)
  ✗ NEVER mode-specific surrogate (R87)

Reference targets (R91 demo benchmarks):
  Steering:  worst +6 dB, ripple 14 dB (no flat-top)
  Flat-top:  worst +0.3 dB, ripple 1.4 dB, 50-70% achievement
  Off-axis:  similar to broadside
```

## 結論

**91 rounds RIS playground 完整 saturate**。所有 hypothesis 都已 tested:
- Positive findings codified in `PATCH_METHODOLOGY.md`
- Negative findings 都有 concrete remedy
- End-to-end demo 確認 pipeline 真的 deliver 「main beam 整片貼上蓋」

**Patch transition methodology 完成 closure**。下一階段建議 patch antenna 實際資料收集。
