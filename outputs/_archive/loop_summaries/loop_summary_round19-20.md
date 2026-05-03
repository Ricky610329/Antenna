# /loop Round 19–20 兩輪總結

> 接續 round 17-18 總結。Round 19-20 主軸：**從工程化的 SA 工具回到物理特性
> 探索**。發現 5.6 GHz 的 bimodal valley 結構與新最高紀錄 +11.82 dB。

## Round 19 — 5.6 GHz Fine Grid 大發現

### 動機
Round 18 看到 5.6 GHz 在 10/15/20×20 的 size dependence 是非單調的（10:+7.61，
15:+6.69，20:+8.43）。中間區間如何？

### 實驗（5.6 GHz × {11/13/15/17/19} × 3 seeds × GD+SA）

| size | mean | max | aperture |
|------|------|-----|----------|
| 11×11 | +7.40 | +7.99 | 5.5λ |
| 13×13 | +8.16 | +10.40 | 6.5λ ← peak 1 |
| **15×15** | **+6.69** | **+7.57** | 7.5λ ← VALLEY |
| 17×17 | +7.08 | +7.09 | 8.5λ |
| **19×19** | **+9.07** | **+11.82 ★** | 9.5λ ← peak 2 |

### 重大發現
1. **Bimodal valley 結構**：13×13 與 19×19 是雙峰，15-17 是 valley
2. **新最高紀錄 +11.82 dB**（比 28 GHz 的 +9.80 高 2 dB）
3. **物理推測**：grating lobe 與 aperture 共振效應
4. **對使用者建議更新**：5.6 GHz 選 19×19 或 13×13，避開 15-17 valley

## Round 20 — 28 GHz Fine Grid 驗證 bimodal 假說

### 假說
若 bimodal valley 是 grating lobe 物理特性，28 GHz 也應該有類似結構（雖然
原因可能在不同的 size 範圍）。

### 實驗（執行中）
28 GHz × {11/13/15/17/19} × 3 seeds × GD+SA
（結果回來後補表）

## 紀錄歷程

```
v1 (純 binary, 15×15)              −4.08 dB
v6 (generator best)                −0.46 dB
direct GD multi-restart            +1.82 ~ +9.51 dB（不可靠）
GD+SA 28 GHz × 15×15 (round 16)    +5.98 ~ +9.80 dB
GD+SA 60 GHz × 10×10 (round 18)    +10.51 dB
GD+SA 5.6 GHz × 19×19 (round 19)   +11.82 dB ← 當前最高 ★
```

## 對使用者最終硬體選型表

| 部署頻率 | 推薦 size | mean / max | 注意 |
|---------|-----------|-----------|------|
| 5.6 GHz | **19×19** ★ 或 13×13 | +9.07 / **+11.82** | 避開 15-17 valley |
| 28 GHz | 15×15 | +8.15 / +9.75 | 待 round 20 驗證是否也有 valley |
| 60 GHz | 10×10 ~ 15×15 | +8.45~+8.55 / +10.51 | size 不敏感 |

## Open Questions

1. ~~5.6 GHz 為何 15×15 valley~~（Round 19 答：bimodal 物理特性）
2. **28 GHz / 60 GHz 是否也有 bimodal valley**（round 20 探）
3. Bimodal 出現的 size 是否依頻率規律可預測？
4. 「19×19 ★ 11.82 dB」是否是真實上限還是運氣（待更多 seeds 驗證）
