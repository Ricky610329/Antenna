# /loop Round 45–46 兩輪總結

> 接續 round 43-44 總結。Round 45-46 完成跨頻率 width sweep，找各頻率的
> main lobe 匹配寬度。

## Round 45 — 60 GHz × 15 Width Sweep（best=60）

60 GHz × 15 × broadside × inc=+60° × 5 restart × SA-per-restart：

| width | 5.6 GHz × 19 | 60 GHz × 15 |
|-------|--------------|-------------|
| 20 | +8.65 | +7.64 |
| 30 | +8.65 | +8.74 |
| **46** | **+11.82 ★** | +9.91 |
| **60** | +9.07 | **+10.14 ★** |
| 80 | +9.62 | +9.67 |

### 重要發現
- **60 GHz × 15 best width = 60**（不是 46）
- 60 GHz × 15 broadside 新最佳 +10.14 dB（vs R40 baseline +9.91, 改善 +0.23）
- 不同 freq × n 配置有自己的 main lobe 匹配寬度

## Round 46 — 28 GHz × 13 Width Sweep（執行中）

完成後三頻率 width 圖譜：
- 5.6 GHz × 19 (9.5λ): width=46 best
- 28 GHz × 13 (6.5λ): TBD
- 60 GHz × 15 (7.5λ): width=60 best

### 待補完整對照表

## 跨頻率 Sweet Configuration（截至 R45）

| Frequency | Best n (aperture) | Best width | Best target | Max suppression |
|-----------|-------------------|-----------|-------------|-----------------|
| 5.6 GHz | 19 (9.5λ) | 46 (~23°) | broadside | **+11.82 ★** |
| 28 GHz | 13 (6.5λ) | TBD | +45.5° | +10.11 (R39 width=46) |
| 60 GHz | 15 (7.5λ) | 60 (~30°) | broadside | **+10.14 ★** (R45) |

## Triple Sharp Peak Confirmed (5.6 GHz)

**+11.82 dB physical record requires ALL**:
1. 5.6 GHz frequency
2. 19×19 size (9.5λ aperture)
3. inc_θ=+60° (sharp peak ±5° → -1.5~-3 dB)
4. broadside target (idx 154-200)
5. width=46 (~23° main lobe match)
6. seed=0 (lucky GD init, 10% 命中)

## Epistemic 進展（截至 R44）

| Round | 假說 | 結果 |
|-------|------|------|
| R12 | ±60° inc 最佳 | R34/42 確認 sharp peak |
| R32 | +11.82 universal | broadside-specific |
| R33 | specular-avoidance | chaotic |
| R35 | best GD → SA | 修為 SA-per-restart |
| R37 | +5.16 mean 真實 | +9.67 with new logic |
| R39/40 | aperture 主導 | 確認 |
| R41 | 越大越好 | 否定 (9.5λ sweet) |
| R42 | inc broad plateau | 否定 (sharp peak) |
| R43 | width 影響小 | 否定 (sharp peak) |
| R44 | 更多 restart 提升命中率 | 否定 (1/10 不變) |
| R45 | width=46 universal | 否定 (60 GHz × 15 best=60) |

## 累計（46 rounds, 75+ commits）

- 17+ scripts: 3 design / 6 sweep / 4 benchmark / 多 diagnostics
- 3 層完整文檔 + 27 round summaries

### 紀錄歷程
```
v1                              −4.08 dB
v6 generator best               −0.46 dB
GD+SA reheat=2 (R25)            +9.75 dB
SA-per-restart 5.6 GHz × 19     mean +9.67, max +11.82
SA-per-restart 60 GHz × 15 (R45) max +10.14
+60° broadside                  +11.82 dB ★ 物理上限保持
```

從 v1 到 broadside 物理上限 = **15.9 dB** 改善

## Open Questions

1. **28 GHz × 13 best width**（R46 探）
2. GPU-batched SA 加速
3. Array factor 數學：能否預測 main lobe 寬度與 best width 對應
