# /loop Round 63 — 最終突破：+30.99 dB

> Loop 結束。R63 推 38 GHz × n=41 達 **+30.99 dB**，96% 理論 array gain。

## TL;DR

| Round | 紀錄 | 配置 |
|-------|------|------|
| R57 | +21.31 | 28 GHz × n=13 |
| R59-61 | +23.02 → +24.97 | 38 GHz × n=15 → 25 |
| R62 | +27.45 | 38 GHz × n=31 |
| **R63** | **+30.99 ★** | **38 GHz × n=41** |

從 v1 (-4.08) 到 R63 (+30.99) = **+35.07 dB 累計改善**

## R63 結果

**38 GHz × n=35/41 × 5 seeds × free-phase + SA**：

| n | aperture | best | mean |
|---|----------|------|------|
| 35 | 17.5λ | +28.32 (seed 3) | +27.07 |
| **41** | **20.5λ** | **+30.99 (seed 0) ★** | **+28.39** |

**28 GHz × n=29/31**（universality 確認）：

| n | best | mean |
|---|------|------|
| 29 | +27.39 | +25.97 |
| **31** | **+28.75** | **+27.74** |

## 接近理論上限

| n | aperture | binary best | theoretical | efficiency |
|---|----------|-------------|-------------|------------|
| 11 | 5.5λ | +15.51 | +20.83 | 74% |
| 21 | 10.5λ | +23.88 | +26.42 | 90% |
| 31 | 15.5λ | +27.45 | +29.83 | 92% |
| **41** | **20.5λ** | **+30.99** | **+32.26** | **96%** |

n=41 達 96% of theoretical array gain，1-bit quantization loss ~1.27 dB。
進一步擴大 aperture 將遇到實際物理製造邊界（~20 cm 板子）。

## 視覺化

四張圖（已存於 `outputs/`）：
1. `best_record_38ghz_n41.png`: 41×41 binary pattern + 響應曲線 + sidelobe distribution
2. `record_progression.png`: v1 → R63 紀錄演進柱狀圖
3. `aperture_scaling.png`: 38 GHz aperture vs suppression vs theoretical
4. `method_comparison.png`: sigmoid vs free-phase 跨 4 頻率對照

## 累計（63 rounds, 100+ commits）

完整 epistemic 鏈：
- R1-R30: explore generator path → conditioning failure
- R30-R56: sigmoid GD + SA 路線優化（卡 +13.44 dB ceiling）
- R55: 文獻 connection 觸發演算法重評估
- **R57-R63: free-phase 演算法替換 → +13.44 → +30.99（+17.55 dB algorithmic）**

最大教訓：
> 演算法層的 attraction landscape 比 hyperparameter sweep 重要 10×。
> R47-R56 9 rounds 認為 +13.44 是 binary ceiling，是 sigmoid path-specific 結論。
> 換 free-phase 立即 +7-11 dB universal improvement，再加 large aperture
> 推到 +30.99（96% theoretical）。
