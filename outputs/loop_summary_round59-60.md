# /loop Round 59–60 兩輪總結 — 兩個 NEW GLOBAL RECORDS

> 接續 round 57-58 的演算法突破。R59 跨頻率 universality 又找到更高 peak，
> R60 進一步 n sweep 再破紀錄。

## TL;DR

| Round | 紀錄 | 配置 |
|-------|------|------|
| R57 | +21.31 dB | 28 GHz × 13 × +51° × width=80 |
| R59 | +23.02 dB | 38 GHz × 15 × +51° × width=80 |
| **R60** | **+23.88 dB ★** | **38 GHz × 21 × +51° × width=80** |

從 v1 (-4.08) 到 R60 (+23.88) = **+27.96 dB 累計改善**

## Round 59 — Free-Phase Universality

對其他頻率測試 R57 free-phase 演算法是否 universally 改善：

| Configuration | Old (sigmoid+SA) | New (free-phase) | Δ |
|---------------|------------------|------------------|---|
| 5.6 GHz × 19 | +11.82 (R37) | +19.61 (R58) | +7.79 |
| 28 GHz × 13 | +13.44 (R47) | +21.31 (R57) | +7.87 |
| **38 GHz × 15** | +11.59 (R53) | **+23.02 (R59) ★** | **+11.43** |
| 60 GHz × 15 | +10.52 (R50) | +17.26 (R59) | +6.74 |

**重要發現：38 GHz 在 free-phase 路線下成新 sweet sweetest**

之前 R47-R56 認為 28 GHz 是 sweet sweetest，是 sigmoid path-specific 結論。
free-phase 路線下：
- 38 GHz × 15 達 +23.02 dB（最高）
- 28 GHz × 13 達 +21.31 dB
- 5.6 GHz × 19 達 +19.61 dB
- 60 GHz × 15 達 +17.26 dB

## Round 60 — 38 GHz × n Sweep

5 seeds × free-phase + SA × 38 GHz × +51° × width=80 × broadside：

| n | aperture | best | mean |
|---|----------|------|------|
| 11 | 5.5λ | +15.51 | +12.78 |
| 13 | 6.5λ | +18.12 | +17.44 |
| 15 (R59) | 7.5λ | +23.02 | +18.91 |
| 17 | 8.5λ | +20.65 | +19.27 |
| 19 | 9.5λ | +21.69 | +20.97 |
| **21** | **10.5λ** | **+23.88 ★** | **+22.55** |

n=21 mean +22.55，**所有 5 seeds 都 ≥+20.90** → robust，不是 lucky。

**新 global record**：38 GHz × n=21 × +51° × width=80 × broadside × seed=3
= **+23.88 dB**

## 跨配置 Sweet Aperture 規律反轉

sigmoid path 與 free-phase path 偏好不同的 sweet aperture：

| Frequency | sigmoid 最佳 n | free-phase 最佳 n |
|-----------|----------------|-------------------|
| 28 GHz | 13 (R51) | TBD（n=13 R57=+21.31，n>15 待測）|
| **38 GHz** | 15 (R53) | **21 (R60) ★** |

物理解讀：
- sigmoid 半圓限制下，更多元素帶來 phase aliasing → 卡 local optimum
- free-phase 全圓下，larger aperture 給更多 phase DoF → 更易達高 suppression

## 演算法 vs 物理層 epistemic 反思

R47-R56 9 rounds 結論「+13.44 是 binary ceiling」**完全錯誤**：
- 真實 ceiling 在演算法層（sigmoid attraction basin）
- 換成 free-phase 立即 +7.87 dB（28 GHz）甚至 +11.43 dB（38 GHz）

R51 結論「n=13 是 28 GHz sweet aperture」**path-specific**：
- sigmoid 下確實 n=13 最佳
- free-phase 下偏好可能更大（38 GHz × 21 = 10.5λ 比 28 GHz × 13 = 6.5λ 大很多）

啟示：**Hyperparameter sweep 結論需要在新演算法下重新驗證**。

## 累計（60 rounds, 95+ commits）

### 工具庫
20+ scripts:
- 3 design tools (sigmoid path)
- 2 NEW: continuous_vs_binary_eval, verify_free_phase_record (free-phase path)
- 6 sweep tools
- 4+ benchmark tools
- 3 層完整文檔 + 33 round summaries

### 紀錄歷程完整鏈
```
v1                                          −4.08 dB
v6 generator best                           −0.46 dB
GD+SA reheat=2 (R25)                        +9.75 dB
SA-per-restart 5.6 GHz × 19 (R37)          +11.82 dB
SA-per-restart 28 GHz × 13 (R48)           +13.44 dB
Free-phase 28 GHz × 13 (R57)               +21.31 dB
Free-phase 38 GHz × 15 (R59)               +23.02 dB
Free-phase 38 GHz × 21 (R60)               +23.88 dB ★ NEW GLOBAL RECORD
```

## Open Questions（更新）

1. n=23/25/27 at 38 GHz 是否再突破 +23.88？（R61 探）
2. 28 GHz × n=15/17/19/21 free-phase 是否也偏好大 n？
3. 24/30/35/45 GHz 等中間頻率是否有 hidden peaks？
4. inc / width fine grid 在 free-phase 下需要重新跑（path-specific sweet）
5. Free-phase + multi-restart 5-10 並行能否更快收斂到全局最佳？
6. Continuous Re/Im parameterization 是否再優於 phase？
