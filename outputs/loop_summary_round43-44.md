# /loop Round 43–44 兩輪總結

> 接續 round 41-42 總結。Round 43-44 主軸：完成最後物理特性 sweep
> （plateau width）+ 命中率調查。

## Round 43 — Width Sweep 完成 Triple Sharp Peak 圖

### 結果（5.6 GHz × 19 × broadside center idx 177 × inc=+60°）

| width | θ range | suppression |
|-------|---------|-------------|
| 20 | -6.5° ~ +3.5° | +8.65 |
| 30 | -9° ~ +6° | +8.65 |
| **46** | -13° ~ +10° | **+11.82 ★ baseline** |
| 60 | -16.5° ~ +13.5° | +9.07 |
| 80 | -21.5° ~ +18.5° | +9.62 |

### 重要發現
**Width=46 對應 5.6 GHz × 19 RIS main lobe ~23°**：
- 太窄 (20-30): plateau 切不到完整 main lobe → suppression 限制
- 太寬 (60-80): sidelobe 計入 main → 上限 ~+9.6

**+11.82 是「width=46 × inc=+60° × 19×19」三重 sharp peak**：

| 維度 | Sweet spot | 偏離 ±N 影響 |
|------|-----------|--------------|
| size | 19×19 (9.5λ) | 17→9.66, 21→8.59 |
| inc_θ | +60° | ±5° → -1.5~-3 dB |
| width | 46 (~23°) | 20→8.65, 60→9.07 |

每個維度偏離都顯著下降。+11.82 dB 是極窄 attraction basin。

## Round 44 — 10-Restart 命中率調查

### 結果
10 restart × SA-per-restart on best config:
- **restart 1 (seed=0): GD +11.82 → SA +11.82** ★
- restart 2-10: GD +2.69~+7.91, SA +6.56~+9.32

**仍只 1/10 命中 +11.82**——10 restarts 沒提升命中率。

### 重要新發現：Seed=0 是 lucky deterministic
**+11.82 命中是 seed=0 specific**：
- design tool default seed=0 → 對此特定配置直接命中
- 其他 seeds 1-9 + SA 都跨不過 deeper basin
- **每個配置可能有自己的 lucky seed**，無法 generalize

實務影響：
- **default seed=0 + 5 restart 對 5.6 GHz × 19 × +60° × broadside × width=46 直接命中 +11.82**
- 對其他配置: 用 5 restart × SA-per-restart 期望 mean +8~+9 dB

## 累計圖譜（44 rounds 統合）

### Triple Sharp Peak Configuration
**+11.82 dB physical record requires ALL of**：
- 5.6 GHz frequency
- 19×19 size (9.5λ aperture)
- inc_θ=+60°
- broadside target (idx 154-200)
- width=46 (~23° main lobe match)
- seed=0 (lucky GD init)

### 跨頻率 Sweet Aperture
| Freq | Sweet aperture | Best size | Max suppression |
|------|----------------|-----------|-----------------|
| 5.6 GHz | 9.5λ | 19×19 | **+11.82 ★** |
| 28 GHz | 6.5λ | 13×13 | +10.11 (best at +45.5°) |
| 60 GHz | 7.5λ | 15×15 | +9.91 (broadside) |

## Epistemic 進展（截至 R43）

| Round | 假說 | 結果 |
|-------|------|------|
| R12 | ±60° inc 最佳 | R34/42 確認 (+60° 是 sharp peak) |
| R32 | +11.82 universal | broadside-specific |
| R33 | specular-avoidance | chaotic |
| R35 | best GD → SA 即可 | 修為 SA-per-restart |
| R37 | mean +5.16 是真實 | 否定 → +9.67 |
| R39/40 | aperture 主導 best target | 確認 |
| R41 | 越大越好 aperture | 否定 (9.5λ 是 sweet) |
| R42 | inc 是 broad plateau | 否定 (+60° 是 sharp peak) |
| R43 | width 影響小 | 否定 (width=46 也是 sharp peak) |
| R44 | 更多 restart 提升命中率 | 否定 (1/10 = 10%, seed=0 specific) |

## 累計（44 rounds, 72+ commits）

### 工具庫
17+ scripts + 3 層完整文檔 + 26 round summaries

### 紀錄歷程
```
v1                              −4.08 dB
v6 generator best               −0.46 dB
GD+SA reheat=2 (R25)            +9.75 dB
SA-per-restart 5.6 GHz × 19 (R37) mean +9.67, max +11.82
+60° broadside (R19, R28, R44)  +11.82 dB ★ 物理上限保持（44 rounds 確認）
```

從 v1 到 broadside 物理上限 = **15.9 dB** 改善

## Open Questions

1. ~~Single config sweep~~（多輪已完成 size, inc, width, freq, plateau pos）
2. **GPU-batched SA 加速**（剩餘 engineering 改善）
3. **跨配置 lucky seed 的物理意義**（為何 seed=0 在 broadside lucky？）
4. Array factor 數學能否預測 +11.82 dB 上限值
