# /loop Round 154–155 兩輪總結 — Multi-frequency joint optimization + BW limit

## TL;DR

Phase 3 patch transition 卡在 HFSS access。Pivot 回 RIS playground 做最 patch-relevant
的測試：**多頻段 joint optimization**，模擬 patch antenna 的 bandwidth 需求。

- **R154**：3 個 in-band freqs (36/38/40 GHz, ~10% rel BW) 一起優化 → **universally
  better than single-freq**（甚至在 design freq 也更好 +0.46 dB）
- **R155**：拓寬 BW 找邊界 → **10% PASS, 32% FAIL（flat-top 崩潰）, 53% FAIL**

Methodology 對標準 patch BW (5-10%) clean PASS。30%+ BW 需要 bigger aperture 或
放鬆 flat-top criterion。

## R154 — Multi-Freq Joint Beats Single-Freq

設定：n=51, inc=51°, w=10° broadside。Loss = sum over freqs of R119 recipe。

| Mode | freq | best | mean | min | flat |
|------|------|------|------|-----|------|
| single-freq @38 | 36 | +2.94 | +0.80 | -0.85 | **4/5** ❌ |
| single-freq @38 | 38 | +2.99 | +1.66 | +0.69 | 5/5 |
| single-freq @38 | 40 | +1.62 | +0.98 | +0.32 | 5/5 |
| **multi-freq joint** | **36** | +2.54 | **+2.19** | **+1.20** | **5/5** ★ |
| **multi-freq joint** | **38** | +2.55 | **+2.12** | +0.94 | **5/5** ★ |
| **multi-freq joint** | **40** | +2.39 | **+1.91** | +0.66 | **5/5** ★ |

### 關鍵發現

1. **Single-freq @38 在 36GHz 退化**：mean 從 +1.66 → +0.80，一個 seed 直接 fail (-0.85)
2. **Multi-freq 全 freq 都 mean ~+2.0+**：完全沒有 off-band degradation
3. **Multi-freq 在 38GHz 比 single-freq 更好**：+0.46 dB 改善！
4. **Bandwidth gain**: off-band mean 改善 +1.16 dB 透過 jointly optimize

Joint optimization 的「regularization 效應」：把 optimizer 困在多頻共識的解，
結果不只更 robust 也比單純 single-freq 解更好。對應 patch transition：
multi-freq spec 不只是 cost，是更穩定的 optimization signal。

## R155 — Bandwidth Limit

| BW | freqs (GHz) | per-freq means | flat | verdict |
|----|-------------|----------------|------|---------|
| ~10% | 36, 38, 40 | +2.44, +2.47, +2.27 | 3/3, 3/3, 3/3 | **PASS** |
| ~32% | 32, 38, 44 | +1.36, +2.01, +2.18 | 1/3, 1/3, 2/3 | **FAIL** flat-top |
| ~53% | 28, 38, 48 | +1.51, +1.19, +1.64 | 1-2/3 | **FAIL** |

### 解讀

10% BW clean PASS — methodology covered。32% 開始 worst-case 仍 positive 但
**flat-top 崩潰**，53% 兩個 metrics 都退化。

物理解釋：每個 binary pixel 同時 contribute 到所有頻率的 response。BW 越寬，
constraints 越多 vs 有限 aperture degrees of freedom。**aperture-vs-BW trade-off**
跟 R125 的 aperture-vs-steering 幾何極限同性質。

### 對 Patch Transition 的 implication

| Patch BW 需求 | RIS methodology | 結論 |
|--------------|-----------------|------|
| 5% (typical narrowband patch) | 完全 cover | 直接 deploy |
| 10% (typical broadband patch) | clean PASS | 直接 deploy |
| 20-30% (UWB) | flat-top 邊界 | 需要 bigger aperture (n > 51) |
| > 30% | recipe re-tune required | 視 application 看是否值得 |

Methodology 對標準 patch deployment scenarios 直接 transferable。

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R154 | Multi-freq joint > single-freq universally; +1.16 dB BW gain |
| R155 | BW limit found: 10% PASS, 32%+ FAIL flat-top |

## 下一階段

R156+ 可選方向：
1. **R156**: Multi-freq + n=71 — 看 bigger aperture 是否破 32% BW boundary
   （類似 R127 對 +45° steering 的 aperture rescue）
2. **R157**: Multi-freq cross-axis robustness — at multiple inc angles, confirm
   methodology still PASS
3. **R158+**: 等 HFSS access 開放，正式進 patch deployment

## 結論

R154-R155 把 RIS playground 推到「patch-equivalent broadband optimization」
場景。發現：
- Loss summed across freqs 的方法直接 work
- Multi-freq 比 single-freq robust **而且** 更好
- 10% BW 在 n=51 clean PASS，是標準 patch 規格
- 30%+ BW 需要 architectural 調整

對「為 patch antenna 建立可信賴方法論」目標：**broadband adaptation 已驗證**，
直接套到 patch 多頻 spec 應 transfer 順利。Phase 3 的 HFSS data 收集還是要做，
但 methodology 已 cover patch 主要應用場景的 BW 範圍。
