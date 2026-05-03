# /loop Round 37–38 兩輪總結

> 接續 round 35-36 總結。Round 37-38 主軸：用新 SA-per-restart 邏輯完成
> plateau 位置物理上限的完整圖譜。

## Round 37 — 7 Plateau Positions Sweep

### 動機
Round 32 發現 +11.82 是 broadside-specific（其他 plateau 只 +0.6~+6.7 dB）。
但那是 GD only seed=0 結果。**用 round 35 修正後的 SA-per-restart 邏輯**，
其他 plateau 位置應該也能改善。

### 實驗（執行中，6/7 完成）
7 plateau positions × 5 restart × SA-per-restart × 5.6 GHz × 19×19 × inc=+60°

| target name | plateau idx | θ_center |
|-------------|-------------|----------|
| left | 91-137 | -33° |
| center_left | 122-168 | -19° |
| broadside | 154-200 | -1.5° |
| center_right | 185-231 | +14° |
| right | 217-263 | +30° |
| far_right | 248-294 | +46° |
| rightmost | 280-326 | +61.5° |

（結果回來後補表）

## Round 38 — 待完整 sweep + 兩輪總結

### 期待
- broadside 仍 +11.82（已知物理上限）
- 其他 plateau 用新 SA-per-restart 邏輯應比 round 32（GD only）顯著提升
- 完整 mapping 給使用者明確硬體選型決策

## 累計（38 rounds, 61+ commits）

### Epistemic 進展鏈（重要假說否定史）
| Round | 假說 | 結果 |
|-------|------|------|
| R12 | ±60° inc 最佳 | R34 否定 → ±30° broadly best |
| R21 | 17×17 universal valley | R22 否定 → 隨 inc_θ 移動 |
| R28 | +11.82 是 lucky 命中 | R31 修正 → reproducible 物理上限 |
| R32 | +11.82 universal | R32 修正 → broadside-specific |
| R33 | specular-avoidance 假說 | R33 否定 → chaotic |
| R35 | best GD → best SA | R35 否定 → 修為 SA-per-restart |
| R36 | 新邏輯破 +11.82 | R36 否定（仍 +11.82） |

### 工具庫總覽
- 3 design tools (with SA-per-restart, --freq, --inc_theta)
- 6 sweep tools (physical_limit, element_num, incidence, freq×size, 2D inc×size, inc_phi)
- 4 benchmark tools (gd_vs_sa, sa_cross_size, sa_reheat, sa_aggressive, test_determinism)
- 多個 diagnostics
- 3 層完整文檔 + 21 round summaries

### 紀錄歷程
```
v1                              −4.08 dB
v6 generator best               −0.46 dB
GD multi-restart                +1.82 ~ +9.51 dB
GD+SA reheat=2 (R25)            +9.75 dB
GD+SA SA-per-restart (R35)      +9.69 dB (inc=+30° broadside)
broadside +60° (R19/28/36)      +11.82 dB ← 物理上限保持 ★
```

從 v1 到當前最佳 = **15.9 dB 改善**

## Open Questions

1. ~~多輪 hypothesis~~（一系列已否定）
2. **完整 plateau × 配置 max suppression 圖譜**（R37 探）
3. GPU-batched SA 加速
4. Array factor 數學能否預測 chaotic patterns
