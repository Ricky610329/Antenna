# Patch Transition Bridge Plan (R151) — SUPERSEDED

> ⚠️ **這份文件已過時, 不要當有效 plan**。寫於 audit `antenna/training/trainer.py`
> 之前, 假設 patch 要從零建 per-task GD pipeline. 真實狀況是 lab 早已有
> amortized G + online learning pipeline. 看新版:
> **`outputs/INTEGRATION_WITH_LAB_PIPELINE.md`**
>
> 留檔目的: 紀錄 R151 當下對「patch transition」的 (錯誤) 理解, 提醒不要
> 沒 audit 既有 codebase 就寫 plan.
>
> ---

> Concrete plan for adapting the R141/R150 RIS pipeline to run on patch
> antenna (HFSS surrogate-in-the-loop). Written 2026-05-03.

---

## What's already in the codebase

| Component | Location | Status |
|-----------|----------|--------|
| `HFSSNet` 6-layer MLP surrogate | `antenna/models/surrogates/hfss_net.py` | Available, 3.98M params at 25×25 input |
| `SurrogateModel` training wrapper | `antenna/models/surrogates/surrogate_model.py` | Available, supports batch training |
| `PatchSimulator` abstract base | `antenna/patch/patch_simulator/__init__.py` | Windows-only (HFSS COM) |
| `SinglePortSimulator` (25×25 pixel) | `antenna/patch/patch_simulator/single_port.py` | Concrete impl |
| `DualPortSimulator` | `antenna/patch/patch_simulator/dual_port.py` | Concrete impl |
| `EnhancedHFSSUNet` (alternative arch) | `antenna/models/surrogates/unet.py` | Available |

**Default response shape**: `(3, 17)` = 3 ports × 17 frequency bins. This is
**S-parameters** (return loss / coupling), not far-field response.

---

## Key differences from RIS pipeline

| Aspect | RIS (Phase 1-2) | Patch (Phase 3) |
|--------|-----------------|-----------------|
| Output | far-field 1D response (361 angles) | S-parameters (3 ports × 17 freqs) |
| "Main beam region" | angular cap around main direction | frequency band where return loss should be low |
| "Sidelobes" | other angular bins | other frequency bins (or off-band response) |
| Forward function | analytical sim (~ms) | HFSS (~minutes) — surrogate mandatory |
| Gradient | through analytical | through surrogate only |
| Truth eval | analytical sim | HFSS COM (called sparingly, e.g. final pattern only) |
| Pixel grid | n × n (n=31/51/71) | 25 × 25 (HFSSNet default) |
| Phase | 0/π binary | binary 0/1 (no phase concept) |

The "1-bit only" constraint maps naturally — patch pixels are also binary
(metal vs no-metal).

---

## Loss adaptation: from RIS to patch S-parameters

RIS recipe was:
```
loss = -(soft_min(main) - soft_max(side))    # worst-case main vs side
     + rw * (soft_max(main) - soft_min(main)) # ripple penalty (main flat-top)
     + lambda * side.mean()                   # area penalty (sidelobe distribution)
```

For patch S11 (return loss, dB), with target band [f_lo, f_hi]:
- "main" = S11 at frequencies in [f_lo, f_hi] — should be **as negative as possible** (good match)
- "side" = S11 outside [f_lo, f_hi] — should be **as positive as possible** (no spurious resonances)

Adapted loss:
```
in_band = S11[f_lo_idx : f_hi_idx]
out_band = S11[outside]
worst = soft_max(in_band) - soft_min(out_band)   # in-band peak vs out-band dip
                                                  # want negative
loss = worst                                    # minimize the gap
     + rw * (soft_max(in_band) - soft_min(in_band))    # in-band flat
     + lambda * out_band.mean()                       # push out-band UP (less interference)
```

(For multi-port, sum/max over ports.)

The same recipe selector idea applies: `(rw, lambda)` chosen by patch
geometry / frequency band / number of ports.

---

## Step-by-step bridge plan (R152+)

### R152 — Wire the API

Generalize `optimize_ris_1bit()` from R150 into `optimize_patch_1bit()`:
- Replace `RISSimulator` with `HFSSNet` surrogate (or EnhancedHFSSUNet)
- Adapt loss to S11 (or chosen response metric)
- Keep recipe selector + joint early-stop structure

Acceptance: function runs end-to-end with a randomly-initialized HFSSNet
(produces garbage output but verifies the wiring).

### R153 — Train HFSSNet on cached data

The codebase has `train_single.py` / `train_dual.py` — these likely train
HFSSNet on existing patch datasets in `result/` directory.
- Identify available patch datasets (probably `result/dataset_v*/`)
- Train HFSSNet to ≥0.85 R² on held-out test set
- Save trained surrogate

### R154 — End-to-end patch pattern via R141 pipeline

Use trained HFSSNet as surrogate in `optimize_patch_1bit()`. Run for one
target frequency band, get a 25×25 binary patch pattern.

Acceptance: pattern's surrogate-predicted S11 satisfies recipe metric
(e.g. in-band S11 < -10 dB).

### R155 — Validate via HFSS COM

Call `SinglePortSimulator(pattern)` once on the optimized pattern. Compare
HFSS truth S11 vs surrogate prediction.

If gap > 3 dB → trigger active learning: add (pattern, HFSS truth) to
training set, retrain surrogate, re-optimize. Loop.

### R156+ — Active learning iterations

Iterate R154-R155 until HFSS truth matches surrogate prediction within
acceptable tolerance.

---

## Risk register

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| HFSS dependency for actual validation | High (Windows-only) | Use `PatchSimulator` only for final validation, not in loop |
| Surrogate fit < 0.85 R² | Medium | R148 showed 0.78 R² still works; if not, more HFSS data |
| HFSS COM stability | Medium | Existing code has `kill()` / `reopen()` for crash recovery |
| Loss mapping non-trivial | Medium-High | Need to study existing patch loss functions in `antenna/losses/` |
| Frequency-domain different from angular | Medium | Same recipe principles apply (worst + ripple + mean) |
| Bandwidth target spec ambiguous | Medium | Need user input on which freq band to target |

---

## Concrete first deliverable

The minimum viable bridge (R152-R155) needs:
1. ~5 min to wire `optimize_patch_1bit()` (R152)
2. Identification of a pre-trained HFSSNet checkpoint OR access to patch dataset (R153)
3. ~30 min for end-to-end optimization (R154)
4. ~5-30 min HFSS validation (R155, depends on availability)

Total: 1-2 hours of focused work, doable inside 3-5 cron cycles.

---

## What R150's findings tell us about R152+

- Even perfect surrogate has 1/4 marginal config (numeric divergence)
- Joint early-stop is **the** safety net; needs eval_fn that's good enough
- For patch, eval_fn = surrogate (HFSS too slow) means joint early-stop
  effectively says "find a snapshot that LOOKS good to surrogate"
- Final HFSS validation closes the loop

This is the standard surrogate-based optimization pattern in EM design — well-
established in literature, and our methodology is now ready to implement it.
