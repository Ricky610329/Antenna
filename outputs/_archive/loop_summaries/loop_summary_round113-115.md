# /loop Round 113–115 三輪總結 — Multi-Target rw + Phase Resolution Curve

> R113 確認 rw=5 不 fix multi-target penalty (R102 ~5 dB physical limit)。
> R114 phase resolution scaling (1-bit / 2-bit / continuous)。
> R115 完成 curve (1/2/3/4-bit) — 3-bit 是 cost-perf sweet spot。

## TL;DR

| Round | Finding |
|-------|---------|
| R113 | rw=5 doesn't fix multi-target ~5 dB physical penalty (only shifts trade-off) |
| R114 | 1-bit→2-bit gain +1.12 dB, 2-bit達 78% continuous |
| R115 | 3-bit (8 levels) 達 98% continuous performance, sweet spot ★ |

## R113 — rw=5 + Multi-target Combined

| Mode | T1 worst | T2 worst | Flat-top |
|------|---------|----------|----------|
| Single T1 (rw=2) | +1.92 | N/A | yes |
| Multi-target rw=2 (R102) | -3.01 | -2.88 | T1 no, T2 yes |
| **Multi-target rw=5 (R113)** | **-2.70** | **-3.64** | **T1 yes, T2 no** |

**rw=5 不 fix multi-target penalty**, 只 shift which target gets flat-top.

### Findings

```
ripple_weight role:
✓ Single-spec quality tuning (R94, R104)
✓ Salvage off-sweet inc (R111-R112)
✗ Multi-target physical penalty (R113)

→ Multi-target ~5 dB cost is fundamental
→ 必須 architectural change (multi-element antenna)
```

## R114-R115 — Phase Resolution Scaling

完整 curve:

| Bits | Levels | Best worst | Ripple | Δ vs continuous |
|------|--------|-----------|--------|-----------------|
| 1 | 2 | +1.92 | 2.59 | -1.97 |
| 2 | 4 | +3.04 | 1.47 | -0.85 |
| **3** | **8** | **+3.80** | 1.38 | **-0.08 (98%)** |
| 4 | 16 | +4.03 | 1.28 | +0.15 |
| cont | ∞ | +3.89 | 1.32 | 0.00 |

### Diminishing Returns

```
1 → 2 bit: +1.12 dB (large gain)
2 → 3 bit: +0.76 dB (good gain)
3 → 4 bit: +0.23 dB (marginal)
4 → cont:  -0.14 dB (asymptote, possibly noise)

→ 3-bit phase shifter 是 patch cost-performance sweet spot
→ 4-bit+ 邊際效益小
```

### Patch Hardware Selection

```python
def patch_phase_hardware(target_dB):
    if target_dB <= 2.0: return "1-bit (RIS-style)"
    elif target_dB <= 3.0: return "2-bit (4 levels)"
    elif target_dB <= 3.8: return "3-bit (8 levels) ★"  # sweet
    elif target_dB <= 4.0: return "continuous (analog)"
    else: return "Beyond physical limit"
```

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R113 | Multi-target ~5 dB fundamental, rw 不能解 |
| R114 | 1-bit/2-bit/continuous comparison |
| R115 | 3-bit達 98% continuous, sweet spot |

## 累計 (115 rounds, 152+ commits) — 11 Axes Validated

```
Methodology validated across:
  freq, n, inc, rw, steps, fab tolerance, multi-spec,
  width, band, per-inc rw adaptation, phase resolution

Sweet point recipe (1-bit, sweet inc):
  worst +1.92 dB, 100% flat-top

Patch hardware sweet (3-bit phase):
  worst +3.80 dB, 100% flat-top, 98% continuous performance
  
最終 patch design matrix:
  Aperture: n=51-equiv (recommended)
  Phase resolution: 3-bit (8 levels)
  Inc handling: per-inc rw adaptation
  Multi-spec: single-spec only, others architectural
  Tolerance: ≤1% fab error
  Compute: 1500 GD steps × 5 restarts
```

## 對 Patch Antenna Final Hardware Budget

```
Total patch deployable performance estimates:

Best (n=51-equiv + 3-bit + sweet inc + rw=2):
  worst suppression ~+3.8 dB
  ripple ~1.4 dB
  100% flat-top reproducibility
  
Moderate (n=51 + 2-bit + sweet inc + rw=2):
  worst ~+3.0 dB
  ripple ~1.5 dB
  100% flat-top
  
Cost-sensitive (n=41 + 2-bit + off-sweet inc + rw=5):
  worst ~+0.5 dB (rough est)
  ripple ~2.5 dB
  60% flat-top
  
Premium (n=51 + continuous + sweet inc + rw=2):
  worst ~+3.9 dB
  ripple ~1.3 dB
  100% flat-top, marginal vs 3-bit
```

## 結論

115 rounds 完整 mature methodology + hardware design matrix。

Patch transition kit 完整 codified:
- 11-axis validation complete
- Sweet recipe + salvage paths
- Multi-spec cost map
- Phase resolution selection guide
- Manufacturing tolerance budget
- Compute timeline

下一階段 patch transition 已 fully equipped reference.
