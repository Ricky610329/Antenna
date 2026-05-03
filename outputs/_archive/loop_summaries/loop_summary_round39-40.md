# /loop Round 39–40 兩輪總結

> 接續 round 37-38 總結。Round 37 用 SA-per-restart 把 5.6 GHz × 19 plateau
> 整個 range 提升 mean +9.67 dB。Round 39-40 跨頻率對比，看是否同樣模式。

## Round 39 — 28 GHz × 13×13 Plateau Full Sweep

### 結果（vs 5.6 GHz × 19）

| target | 5.6 GHz × 19 | 28 GHz × 13 |
|--------|--------------|-------------|
| left (-33°) | +9.44 | +7.87 |
| center_left (-17.5°) | +9.84 | +9.93 |
| **broadside (-1.5°)** | **+11.82 ★** | +9.18 |
| center_right (+14°) | +9.57 | +9.17 |
| right (+30°) | +9.43 | +8.71 |
| **far_right (+45.5°)** | +9.29 | **+10.11 ★** |
| rightmost (+61.5°) | +8.32 | +7.38 |
| **mean** | **+9.67** | **+8.91** |
| **max** | +11.82 | +10.11 |

**意外規律**：兩配置最佳 target 方向不同！
- 5.6 GHz × 19 (9.5λ aperture, 508mm): best 在 broadside
- 28 GHz × 13 (6.5λ aperture, 70mm): best 在 +45.5°

物理：array factor × aperture 多因子 interaction，不是 inc_θ 的簡單函數。

## Round 40 — 60 GHz × 15×15 Plateau Full Sweep（完成）

| target | 5.6 GHz × 19 | 28 GHz × 13 | 60 GHz × 15 |
|--------|--------------|-------------|-------------|
| left | +9.44 | +7.87 | +9.28 |
| center_left | +9.84 | +9.93 | +8.58 |
| **broadside** | **+11.82 ★** | +9.18 | **+9.91 ★** |
| center_right | +9.57 | +9.17 | +9.14 |
| right | +9.43 | +8.71 | +9.25 |
| **far_right** | +9.29 | **+10.11 ★** | +9.48 |
| rightmost | +8.32 | +7.38 | +8.04 |
| **mean** | **+9.67** | +8.91 | +9.10 |

### 規律發現：Aperture × Best target
- 9.5λ (5.6 GHz × 19): **broadside best**
- 7.5λ (60 GHz × 15): **broadside best**
- 6.5λ (28 GHz × 13): **+45.5° best**

→ **Aperture ≥7.5λ 時 broadside 最佳；中等 aperture (6.5λ) 偏側方向最佳**

→ Aperture 大小是主導變因，不是頻率本身。

## Epistemic 進展（截至 R39）

| Round | 假說 | 結果 |
|-------|------|------|
| R12 | ±60° inc 最佳 | R34 否定 → ±30° broadly best |
| R21 | 17×17 universal valley | R22 否定 → 隨 inc_θ 移動 |
| R28 | +11.82 lucky 命中 | R31 修正：reproducible |
| R32 | +11.82 universal | broadside-only |
| R33 | specular-avoidance | chaotic |
| R35 | best GD → best SA | 修為 SA-per-restart |
| R36 | 新邏輯破 +11.82 | 仍 +11.82 |
| **R39** | **5.6 GHz × 19 broadside-best 規律** | **28 GHz × 13 best 在 +45.5°（不同）** |

## 累計（39 rounds, 64+ commits）

### 紀錄歷程
```
v1                              −4.08 dB
v6 generator best               −0.46 dB
GD multi-restart                +1.82 ~ +9.51 dB
GD+SA reheat=2 (R25)            +9.75 dB
SA-per-restart 5.6 GHz × 19 (R37)  mean +9.67, max +11.82
SA-per-restart 28 GHz × 13 (R39)   mean +8.91, max +10.11
broadside +60° (R19)            +11.82 dB ★ 物理上限保持
```

### 工具庫總覽
- 3 design tools (with SA-per-restart fix)
- 6 sweep tools
- 4 benchmark tools
- 多個 diagnostics
- **3 層完整文檔 + 23 round summaries**

## 對使用者最終選型表（R39 為止）

| 部署頻率 | 推薦 size | best 方向 | mean | max |
|---------|-----------|-----------|------|-----|
| 5.6 GHz | 19×19 | broadside (-1.5°) | +9.67 | +11.82 ★ |
| 28 GHz | 13×13 | far_right (+45.5°) | +8.91 | +10.11 |
| 60 GHz | 15×15? | TBD | TBD | TBD |

## Open Questions

1. **60 GHz × 15 best target 方向**（R40 探）
2. 是否有 universal aperture × inc 規律可預測 best target
3. GPU-batched SA 加速
