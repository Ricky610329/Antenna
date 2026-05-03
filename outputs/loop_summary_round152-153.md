# /loop Round 152–153 兩輪總結 — Patch transition plumbing + small-aperture extension

## TL;DR

R152 wires up `optimize_patch_1bit()` plumbing and discovers an existing
pre-trained surrogate checkpoint at `result/_pretrained_surrogate/` — but
it's actually an **n=15 RIS surrogate** (not patch), and is too inaccurate
(6.29 dB mean error) to drive optimization. R153 falls back to R146's
warm-start trick at n=15 → exact match → R141 pipeline runs end-to-end at
new aperture; surrogate-loop matches analytical (delta mean -0.01).

Patch bridge plan needs revision: there's no usable pre-trained patch
surrogate; need to actually train one. But the methodology pieces (selector
+ warm-start + joint early-stop) are now validated across n ∈ {15, 31, 51, 71}.

## R152 — Pipeline Plumbing Test

### Discovery

`result/_pretrained_surrogate/checkpoint/sm.pth` exists with metadata:
```
{
  "element_num": 15,
  "n_samples": 5000,
  "epochs_used": 200,
  "final_loss": 56.37,
  "pattern_size": 225,
  "response_size": 361
}
```

→ This is a **RIS surrogate at n=15**, NOT a patch antenna surrogate as
the directory name suggested.

### Loading + smoke test

- HFSSNet(225, (1, 361)) loads checkpoint cleanly with strict=False
- Inference produces output range -17 to +9 dB (looks plausible)
- BUT: random patches give nearly identical predictions (e.g., 6.29 dB mean
  abs err vs analytical truth, 29.47 dB max)
- Optimization with sigmoid soft-binarize stuck at constant loss for 500 steps
- Surrogate predicts worst=-0.81 (good!) but truth says worst=-12.37 (terrible!)

### Diagnosis

The pre-trained surrogate is **unusable for surrogate-loop optimization**.
Same failure mode as R142-R145: HFSSNet trained on random binary patterns
with MSE loss doesn't capture the |F·pattern|² structure precisely enough
for gradient-based optimization to work.

`final_loss = 56.37` in meta corresponds to ~7.5 dB RMSE — confirms the
surrogate is too coarse to drive optimization.

## R153 — Warm-start Surrogate at n=15

### Approach

Apply R146's proven warm-start trick: extract `sim.pre_calAF[0]` complex
coefficients and copy into `WarmStartSurrogate` weights → exact match
without any training.

```
Warm-start fit (50 random binary patches): mean abs err = 0.000002 dB
```

### Run R141 pipeline at n=15

| Mode | Best | Mean | Min | Flat |
|------|------|------|-----|------|
| Analytical | +0.34 | +0.04 | -0.33 | 5/5 |
| Surrogate | +0.35 | +0.03 | -0.53 | 5/5 |
| delta_mean | -0.01 | | | tie |

### Findings

- **Methodology generalizes to n=15**, outside R141 selector envelope ({31, 51, 71})
- **Surrogate-loop matches analytical** even at this small aperture
- **Joint early-stop** keeps 5/5 flat-top in both modes
- **Practical limit**: n=15 best_worst only +0.34 dB → physical aperture too small
  for production deployment. R141 selector's n_min=31 was correctly chosen.

## Pipeline Now Validated

| Aperture | Analytical | Surrogate | Status |
|----------|-----------|-----------|--------|
| n=15 | ✓ R153 | ✓ R153 | small-aperture extreme |
| n=31 | ✓ R134 | ✓ R148/R149 | validated |
| n=51 | ✓ R134 | ✓ R150 | validated, primary deployment |
| n=71 | ✓ R134 | ✓ R149 | large-aperture, off-normal mmWave |

End-to-end methodology stable across 4× aperture range with both modes.

## Patch Bridge Plan Revision

R151 plan said "R153 = train HFSSNet on existing patch dataset". After R152
discovery, this is wrong on two counts:
1. The "patch dataset" doesn't actually exist (the checkpoint is RIS, not patch)
2. Even if patch data existed, the HFSSNet architecture clearly isn't sufficient
   (R142-R145 lessons + R152 confirmation)

Revised R154+ plan:
1. **R154**: Build a true patch dataset by running PatchSimulator (HFSS COM)
   on N=50-200 random binary patterns. Save (pattern, S-param) pairs.
2. **R155**: Train a physics-aware patch surrogate (analog to R143's design
   but for S-parameters — use SISO transfer function structure)
3. **R156**: Run optimize_patch_1bit() with trained surrogate
4. **R157**: HFSS validation of final pattern, active learning if needed

This is more realistic given findings; R153 confirmed the methodology will
transfer once we have a usable surrogate.

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R152 | Pipeline plumbing OK; existing pre-trained surrogate unusable |
| R153 | Warm-start at n=15 succeeds; methodology validated 4× n range |

## 下一階段

R154+ build actual patch dataset (HFSS-based) before training real patch
surrogate. The Phase 1-2 methodology is solid; Phase 3 needs HFSS time.

If HFSS access is limited (Windows-only, slow), can also:
- Document the bridge as ready-to-use API
- Demonstrate one full pipeline run with synthetic patch data
- Wait for actual HFSS access window

## 結論

兩個 round 一個是 plumbing exploration（R152，發現 existing assets 不夠用），
一個是 small-aperture extension（R153，確認 methodology 跨 4× n range 都 work）。

Pipeline 端到端 ready，等實際 HFSS data 進來就能開始 patch antenna deployment。
methodology 從 R94 baseline 走到 R153，已 validate 在多 aperture × 多 freq ×
多 inc × 多 width × surrogate noise 各種 axis 上。
