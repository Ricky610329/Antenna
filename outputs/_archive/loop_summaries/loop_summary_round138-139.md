# /loop Round 138–139 兩輪總結 — Early stopping 看似贏，generalization 揭露 trade-off

## TL;DR

R138 在 R119 sweet spot 用 early-stopping 修掉 Adam drift（mean_worst -0.38 → +2.79，
no negative seeds）。R139 把同 intervention 套到其他 selector recipes 卻發現
**flat-top compliance 在 3/4 configs 崩潰**。R138 的 win 是 config-specific，
不是 universal upgrade。需要 **joint early-stop criterion**（worst AND flat-top
同時滿足）才能 promote 到 selector default。

## R138 — Early Stopping at R119 Sweet Spot

R137 anomaly（mean_worst at 1500 steps -0.38, 3000 才回升 +2.23）被 reproduce 並修復：

| variant | best | mean | min | flat | per-seed |
|---------|------|------|-----|------|----------|
| baseline (R137) | +3.03 | -0.38 | **-8.02** | 4/5 | -8.02 +0.69 +1.15 +3.03 +1.25 |
| **early-stop only** | **+3.48** | **+2.79** | **+1.32** | 4/5 | +2.59 +3.48 +3.44 +3.09 +1.32 ★ |
| lr-decay only | +2.52 | +1.69 | +0.25 | **5/5** | tighter spread, lower best |
| early-stop + decay | +3.83 | +2.48 | +0.42 | 3/5 | best worst but loses flat-top |

R138 結論（過度樂觀）：early-stopping 是 universal 修掉 Adam drift 的 transferable pattern。

## R139 — Universality 測試 → 翻案

對 4 個 selector recipes 跑 (final-step) vs (early-stop) 比較：

### Config A: R119 narrow (n=51, w=10) — R138 reference
- best_worst Δ +0.45
- min_worst Δ **+9.34** (大 win)
- **flat 維持 4/5** ✓

### Config B: R129 wide (n=51, w=18)
- best_worst Δ +1.37
- min_worst Δ +1.73
- **flat 5/5 → 1/5** ❌ DROPS

### Config C: R131 inc=0° 28GHz (rescue)
- best_worst Δ +0.60
- min_worst Δ +1.48
- **flat 5/5 → 1/5** ❌ DROPS

### Config D: n=71 extrapolation
- best_worst Δ **+1.98**
- min_worst Δ +3.06
- flat 3/3 → 2/3 (掉一個)

### Pattern

```
Early-stop ALWAYS improves worst-case metrics (massive on min_worst)
BUT trades flat-top in 3/4 configs

Why: best worst-case along trajectory often happens BEFORE flat-top
fully establishes. Picking that snapshot maximizes worst but ripple
hasn't been suppressed yet.

R119 (Config A) is exception because it has so much headroom that
both worst and flat-top stabilize together.
```

## 修正：Joint Early-Stop Criterion

R138 的「track best worst-case」太簡單。Better criterion：

```python
# Pseudocode
best_state = None
best_worst_among_valid = -inf

for step in range(gd_steps):
    optimizer step...
    
    if (step+1) % eval_every == 0:
        m = evaluate_binary(params)
        # Only consider snapshots where flat-top is satisfied
        if m["flat_top"] and m["worst"] > best_worst_among_valid:
            best_worst_among_valid = m["worst"]
            best_state = params.detach().clone()

# Fallback: if no valid snapshot found, use final-step
if best_state is None:
    best_state = params.detach()
```

這個「joint criterion」是 R140 應該驗證的下一步。

## 真實的 take-away

1. **R137 mean_worst dip 是真實的問題** — Adam at lr=0.05 在某些 seed 上
   over-optimize side_mean 後 worst-case 就垮
2. **R138 early-stop 修法太天真** — 只 track worst 會犧牲 flat-top
3. **R139 真實 pattern**：早停的 trajectory selection 必須考慮**多 metric joint**
4. 對 surrogate-in-the-loop：trajectory selection 仍是有效 transferable pattern，
   但 criterion 要 align 真實 deployment metric (worst > 0 AND flat-top OK)

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R138 | Early-stop 在 R119 sweet spot 修 Adam drift（看似 universal win）|
| R139 | 4 recipe 測試揭露 early-stop 在 wide-cap / inc=0 配方 trade flat-top |

## 下一階段建議

1. **R140**: 實作 joint early-stop criterion（worst AND flat-top 同時滿足）+ 4 recipe re-validate
2. **R141**: 把 joint early-stop integrate 到 selector，更新 deployment-ready function
3. **R142+**: 進入 surrogate-in-the-loop 階段

## 結論

這兩輪是個「scientifically humbling」週期：
- R138 看起來找到 universal upgrade
- R139 generalization test 翻案
- 真實的 transferable pattern 比想像複雜

對 patch transition 的價值：**這個 trade-off 在 surrogate workflow 也會出現**。
Early-stopping criterion 必須跟 deployment acceptance criterion 一致，
不然 surrogate gradient 收斂到的 pattern 會 fail 真實 fab specs。

Loss design + recipe selector + correct trajectory selection criterion =
真正 deployment-ready 的 methodology。R140 修正後才能真正 promote 到 default。
