# /loop Round 148–149 兩輪總結 — Surrogate-loop robustness 全面驗證

## TL;DR

R146-R147 證明完美 warm-start surrogate 跟 analytical 等價。R148-R149 驗證
**imperfect surrogate** 也 work：

- **R148**：weight noise sweep（0/5/10/20% of std）→ 全部 PASS，且 noise 越大
  mean worst 越**好**（surrogate noise 像 exploration regularization）
- **R149**：10% noise × 4 個 selector configs（不同 n / inc / freq / width）→ **全部 PASS**，
  surrogate 在每個 config 都比 analytical baseline mean **更高**

Phase 2 結論：**methodology 在 imperfect surrogate 上比 perfect surrogate 還好**。
Patch transition 風險已大幅降低，可以進入 Phase 3。

## R148 — Perturbation Sweep at One Config

n=31, inc=51, 38GHz, w=10, R119 recipe，noise 加到 warm-start surrogate weights：

| Noise | R² | Fit err | Best worst | Mean worst | Min worst | Flat-top |
|-------|------|---------|-----------|------------|-----------|----------|
| truth | - | - | +0.95 | +0.66 | +0.33 | 5/5 |
| 0% | 1.0000 | 0.000 dB | +0.82 | +0.68 | +0.56 | 5/5 |
| 5% | 0.9778 | 0.45 dB | +1.38 | +0.84 | +0.33 | 5/5 |
| 10% | 0.9267 | 0.88 dB | +1.25 | **+0.87** | +0.53 | 5/5 |
| 20% | 0.7845 | 1.65 dB | +1.32 | **+0.94** | +0.51 | 5/5 |

**驚人發現**：noise 5/10/20% 都比 truth baseline mean (+0.66) 好！

Theory：surrogate gradient noise 起到 exploration / regularization 效果。
Joint early-stop 用 analytical truth eval，所以 noise-induced 壞 pattern 被
filter 掉。剩下的是 noise-helped escape from local minima 的好 pattern。

## R149 — Cross-Config Generalization at 10% Noise

| Config | Truth mean | Surrogate mean | Δ mean | Flat T/S | Verdict |
|--------|-----------|----------------|--------|----------|---------|
| A: n=31 inc=51 R119 | +0.66 | +0.92 | +0.26 | 5/5 | PASS |
| B: n=51 inc=51 R119 | +2.47 | **+3.05** | +0.58 | 5/5 | PASS |
| C: n=31 inc=51 R129 wide | +0.33 | +0.77 | +0.44 | 5/5 | PASS |
| D: n=31 inc=0 R131 rescue | -1.32 | -0.66 | +0.67 | 4/4 | PASS |

**所有 4 configs PASS**。surrogate-loop 在每個 config 都 beat analytical baseline。

註：D 的 truth mean = -1.32 < 0，因為 R131 recipe 是為 n=51 設計，套到 n=31
本來就 underperform。即便如此 surrogate-loop 還是改善 +0.67 dB。

## Phase 2 完整結論

| Round | 結論 |
|-------|------|
| R142 | CNN 架構不夠表達 \|F·x\|² + log |
| R143 | Physics-aware + random data 仍 cold-start fail |
| R144 | Trajectory data 反而更糟（dynamic range 大）|
| R145 | Warm-start indexing bug (-0.97) |
| R146 | Warm-start fixed → R² = 1.000000 ★ |
| R147 | Continuous-aware surrogate-loop = analytical ★ |
| **R148** | **20% weight noise 都 PASS（比 truth 還好）** |
| **R149** | **10% noise × 4 configs 全 PASS** |

### Patch Transition Risk Re-evaluation

| Risk | 原本 | 現在 |
|------|------|------|
| Methodology transferable? | 大 risk | ✓ R147 確認 |
| Surrogate noise robust? | 大 risk | ✓ R148/R149 確認 (20% 都 OK) |
| Cross-config generalization? | 大 risk | ✓ R149 確認 |
| Joint early-stop 在 imperfect surrogate 仍 work? | 中 risk | ✓ R148/R149 確認 |
| HFSS surrogate fit quality? | 大 risk | TODO Phase 3 |
| Patch antenna geometry effects? | 大 risk | TODO Phase 3 |

**Phase 2 通關**。剩下的 risk 都是「需要實際 patch HFSS data」才能驗證。

## Speed Comparison

| Step | Surrogate-loop | Analytical-loop |
|------|----------------|-----------------|
| n=31 | 28s | 87-89s (3.1x) |
| n=51 | 29s | 225s (7.7x) |

**Surrogate gives 3-8x speedup**。對 patch HFSS 場景每個 sim 是分鐘級，
surrogate-loop 的 speedup 是「能 deploy vs 不能 deploy」的差別。

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R148 | Weight noise sweep — 全部 PASS, surrogate noise 反而 helps |
| R149 | 10% noise × 4 selector configs — all PASS, surr beats truth in all |

## 下一階段建議 (R150+)

進入 Phase 3 patch transition：

1. **R150**: 整合 surrogate-loop 進 R141 deployment function — 讓
   `optimize_ris_1bit()` 接受 optional surrogate 參數
2. **R151**: 看 codebase 已有的 `PatchSimulator` 跟 `HFSSNet` —
   它們能跟 R141 selector + surrogate-loop 接起來嗎？
3. **R152**: 用 patch sim 生少量 data，train HFSSNet surrogate，套用 R141
   pipeline，跑出第一個 patch antenna pattern
4. **R153+**: HFSS validation via COM interface（或先用 patch sim 模擬）

## 結論

Phase 2 從 R142 連續 4 輪 negative 開頭，到 R146 architecture 確認，R147
methodology transfer，R148-R149 robustness 全面 OK，正式收尾。

關鍵 insight：**surrogate-loop 不只是 OK，是 actively better**。surrogate
gradient noise 像 SGD 的 implicit regularization，幫助 escape local minima。
Joint early-stop 是 critical safety net，確保 noise-induced bad pattern 不會
被選中。

Phase 3 風險評估：methodology 已 derisked，剩下的是 patch-specific
data engineering。原計劃多月的 patch transition risk discovery 在 8 輪
RIS playground 實驗中完成。
