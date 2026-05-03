# /loop Round 70–71 兩輪總結 — Symmetry / Augmentation / Dataset Visualization

> R70 驗證 RIS 對稱性 + augmentation 測試。R71 dataset_v1 gallery 視覺化揭露
> 「同 config 多 mode」結構，對 generator 訓練有重要 implication。

## TL;DR

- **Bit-flip invariance** (0↔1) 唯一 RIS symmetry — 其他空間變換 break
- **Augmentation 沒幫助** — MAE 4.95 vs 4.89 (R68)
- **同 config 下 rw=0 vs rw=2 的最優 pattern hamming 51.72%** — 幾乎完全不同
- → (config → pattern) 是 multimodal mapping，generator 必須 conditional on use case

## R70 — Symmetry Verification + Augmentation

### Verified symmetries

```
Bit-flip (0↔1):           INVARIANT (1e-5 dB)  ← 全局相位 π 加成
flipud (x → -x):          changes 19 dB
fliplr (y → -y):          changes 27 dB
transpose:                changes 32 dB
rot90 / rot180:           changes 28 dB
```

只有 bit-flip 保留 response。incidence direction (inc_θ=+51°, inc_φ=90°)
打破所有空間對稱。

### Augmentation 結果

| Setup | MAE worst_supp |
|-------|---------------|
| R68 CNN no augment | 4.89 dB |
| R70 CNN bit-flip augment | 4.95 dB |

**沒有改善**。Model 已 implicitly 學會 symmetry，augmentation 不解決
fundamental sample efficiency 問題。

### 對 patch 移植 lesson

1. 每種 symmetry 須個別驗證（patch 不一定有 bit-flip 等價）
2. 簡單 augmentation 不是 magic bullet
3. Real bottleneck: dataset diversity, not sample count

## R71 — Dataset Visualization

### Gallery (`outputs/dataset_v1_gallery.png`)

36 entries × (rw=0 pattern, rw=2 pattern, response 對比)。
- Binary patterns 看似 quasi-random（無明顯結構）
- rw=0 response: 紅色 sharp peak + 大幅 ripple
- rw=2 response: 藍色 flat-top + 較窄 ripple

### Pareto detail (`outputs/pareto_compare_38GHz_n31.png`)

挑 38 GHz × n=31 × broadside × w=20 detail：

| Mode | worst_supp | ripple | main < -3 dB |
|------|-----------|--------|--------------|
| rw=0 | +2.05 | 13.79 | 26/40 |
| rw=2 | -0.30 | 2.17 | 0/40 (flat-top ✓) |

**Hamming distance = 51.72%** ← 兩 pattern 幾乎完全不同。

## Critical Insight: (config → pattern) 是 Multimodal Mapping

同 physical config（freq, n, target shape, inc）有**至少兩個 disjoint optimal binary patterns**：
- pattern_A: 集中尖峰（max-max 最佳）
- pattern_B: flat-top（worst-case 最佳）

兩者 hamming ~50% = 完全不相關。

### 對 R1-R30 generator failure 的解釋

實驗室原本訓 (target → pattern) NN，所有 target 學到同個 pattern：
- 不是「conditioning fail」單純問題
- 是 (target → pattern) **基本就是 multimodal** mapping
- NN 學 mean → mean of bimodal = 中間 garbage
- + max-max metric overestimates "中間" pattern → 看起來能用

### 對 patch antenna 的啟示

**Generator-style approach 必須 conditional on use case mode：**

```
GOOD:  (config + use_case_mode) → pattern
       e.g. mode = {"steering", "flat_top", "balanced"}

BAD:   (config) → pattern   ← collapses to mean of multiple modes
```

Patch 對應：S11 vs gain 也是 trade-off，不能用單一 generator 涵蓋。

## R71 進度

dataset_v2 (54 configs × 4 = 216 runs，含 n=41) 在背景跑。**36/54** 完成。
預計 30 min 跑完。R72+ 用 v2 重訓 surrogate 看 scaling。

## 紀錄歷程修正（截至 R71）

| 階段 | 焦點 | 結果 |
|------|------|------|
| R57-R63 max-max | single steering | +30.99 (虛胖) |
| R64-R65 worst-case | flat-top deployment | +6.88 honest |
| R66-R67 dataset_v1 | 多 use case 涵蓋 | 36 entries Pareto |
| R68-R69 surrogate POC | forward MAE 4.89 / metric collapse | dense supervision wins |
| **R70 augmentation** | bit-flip invariance | **無改善** |
| **R71 visualization** | gallery + Pareto detail | **多 mode 結構揭露** |

## Open Questions（更新）

1. dataset_v2 (5x v1) surrogate MAE 能否從 4.89 → < 2 dB？
2. **Conditional generator** with explicit mode embedding 能否避開 R1-R30 collapse？
3. Active learning vs random scaling 在 multimodal landscape 下哪個更有效？
4. 是否有 third 中等 mode (rw=1) Pareto 點？hamming 是 50% 三合一還是雙峰？
5. Patch surrogate methodology: 預測 full curve + use case conditioning
