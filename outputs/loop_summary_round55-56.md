# /loop Round 55–56 兩輪總結

> 接續 round 53-54 總結。Round 55 確認 width knife-edge，Round 56 完成
> 8-維度 sharp peak 規律的 seed/aperture 兩維度。

## Round 55 — Width=80 Knife-Edge 確認

28 GHz × 13 × +51° × broadside × width fine grid（補 R46 粗網格）：

| width | suppression | vs 80 |
|-------|-------------|-------|
| 70 | +9.58 | -3.86 |
| 75 | +11.24 | -2.20 |
| **80** | **+13.44 ★** | — |
| 85 | +10.17 | -3.27 |
| 90 | +9.97 | -3.47 |
| 100 | +10.74 | -2.70 |

±5 偏離下降 2-4 dB → **width=80 是 knife-edge sharp peak**。
width=75 / 100 是 secondary local peaks。

## Round 56 — Seeds 5-9 + Larger n=21/23/25

### Seeds 5-9 結果

| seed | suppression |
|------|-------------|
| **0** ★ | **+13.44** |
| 5 | +8.60 |
| 6 | +7.40 |
| 7 | +6.99 |
| 8 | +8.18 |
| 9 | +10.16 ← secondary |

mean of 5-9 = +8.27, max = +10.16。
**1/10 lucky rate 確認**——只有 seed=0 達 +13.44。

### Larger n sweep (n=21/23/25)

| n | aperture | suppression |
|---|----------|-------------|
| **13** | 6.5λ | **+13.44 ★** |
| 21 | 10.5λ | +8.95 |
| 23 | 11.5λ | +9.11 |
| 25 | 12.5λ | +10.68 |

→ 越大 aperture 不能突破 +13.44。**n=13 是全局最佳**。

## 完整 8-維度 Sharp Peak（截至 R56）

| 維度 | Sweet | 偏離影響 |
|------|-------|----------|
| freq | 28 GHz | ±2 GHz: -3 ~ -4 dB (R54) |
| n | 13 (6.5λ) | n=11-25: -2 ~ -5 dB (R51/R56) |
| inc | +51° | ±1°: -3 ~ -4 dB (R48 knife-edge) |
| width | 80 | ±5: -2 ~ -4 dB (R55 knife-edge) |
| plateau pos | broadside | center_left -0.9, right -4.3 (R52) |
| target shape | flat plateau | TBD |
| seed | 0 | seeds 1-9: -3 ~ -7 dB (R56) |
| algorithm | SA-per-restart | -1 ~ -2 dB without (R35) |

## 文獻 Connection（R55 新增）

3 papers 確認我們的發現對應已知 1-bit RIS theory：

1. **3 dB quantization loss**: 1-bit RIS 比 continuous phase 損失 ~3 dB
   → 我們的 +13.44 dB binary 對應 continuous ceiling ~+16-17 dB
2. **Quantization grating lobes**: 1-bit phase 因離散性產生 SLL 限制
3. **Phase randomization / prephased 1-bit metasurface**:
   add random pre-phase 打破 quantization periodicity 降低 QLL
   → **lucky GD seed=0 ≡ implicit phase pre-randomization solution**
     for the specific (freq, inc, width, target) configuration

Multi-restart 是 empirical phase-randomization exploration。
SA-per-restart 在每個 random pre-phase 周圍 local optimization。

## 紀錄歷程（截至 R56，未變）

```
v1                                          −4.08 dB
v6 generator best                           −0.46 dB
GD+SA reheat=2 (R25)                        +9.75 dB
SA-per-restart 5.6 GHz × 19 × +60° (R37)   +11.82 dB
SA-per-restart 28 GHz × 13 × +51° (R48)    +13.44 dB ★ 物理紀錄
```

從 v1 到當前 = **17.52 dB 改善**

## Epistemic 進展（截至 R56）

| Round | 假說 | 結果 |
|-------|------|------|
| R51 | n=11/15 達 +14 | 否定 (n=13 sweet) |
| R52 | +13.44 broadside-only on 28 GHz | 確認 |
| R53 | 不同頻率有 hidden +14 紀錄 | 否定 |
| R54 | 28 GHz 鄰近 ±2 GHz 細網格更高 | 否定 (sweet sweetest) |
| R55 | width 80 周圍細網格更高 | 否定 (knife-edge) |
| R56a | seed≠0 也達 +13.44 | **否定 (truly seed=0 only)** |
| R56b | n>19 大 aperture 突破 +13.44 | 否定 (n=13 全局最佳) |

## 累計（56 rounds, 91+ commits）

- 17+ scripts，3 層完整文檔 + 32 round summaries
- 8 個頻率（5.6/12/24/26/27/28/29/30/38/60 GHz）
- 9 個 sizes (n=11~25 in steps of 2)
- 完整 inc/width fine grid (knife-edge structure)
- 文獻 3 papers grounded：3 dB quantization loss / phase randomization

## 收斂判斷

R47 (+13.44) 之後 9 rounds (R48-R56) 嘗試突破：
- 不同 inc (R48 knife-edge ±1°)
- 不同 plateau pos (R52)
- 不同 freq (R49/R50/R53/R54)
- 不同 width (R55)
- 不同 seed (R56a)
- 不同 n (R51/R56b)

**全部失敗** → +13.44 是 1-bit RIS 對「28 GHz × 13 × +51° × width=80
× broadside × seed=0」配置的 attraction basin upper bound，吻合
文獻 3 dB quantization loss + phase randomization theory。

## Open Questions（更新）

1. **Continuous phase comparison** 驗證 +16-17 dB ceiling（3 dB loss
   theory empirical 確認）
2. **Target shape sweep**（flat plateau vs Gaussian vs ramp）未做
3. **GPU-batched SA** 加速（仍未做）
4. 其他 inc/width/freq 組合是否有 hidden +13+ dB（已大量探索，剩餘可能性低）
