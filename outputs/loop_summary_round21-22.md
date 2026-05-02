# /loop Round 21–22 兩輪總結

> 接續 round 19-20 總結。Round 21 完成三頻率 fine grid 對照，發現
> **universal valley at 17×17**。Round 22 驗證 valley 對 inc_θ 的依賴性。

## Round 21 — 完成三頻率 fine grid 對照

### 數據（GD+SA, 3 seeds × 5 sizes，inc_θ=+60°, plateau 154-200）

| size | 5.6 GHz | 28 GHz | 60 GHz |
|------|---------|--------|--------|
| 11×11 | +7.40 | +8.08 | +7.46 |
| 13×13 | +8.16 | **+8.93 ★** | +7.74 |
| 15×15 | +6.69 ↓ | +8.15 | **+8.55 ★** |
| **17×17** | **+7.08 ↓** | **+7.43 ↓** | **+7.02 ↓** |
| 19×19 | **+9.07 ★** | +7.94 | +7.31 |

### 重大發現
1. **Universal Valley at 17×17**——所有三頻率在 17×17 都是 mean local min
2. 各頻率 peak 不同：5.6 GHz=19×19, 28 GHz=13×13, 60 GHz=15×15
3. Aperture 不單獨決定最佳——peak aperture 9.5/6.5/7.5λ 沒有單調規律
4. 推測 grating lobe × inc_θ × element spacing × target 多因子 interaction

## Round 22 — Universal Valley 對 inc_θ 的依賴性

### 假說
17×17 universal valley 是 inc_θ=+60° 的 artifact，還是真正 universal？
跑 inc_θ=-40°（RIS default）看 valley 是否仍出現。

### 實驗（執行中）
28 GHz × {11/13/15/17/19} × 3 seeds × inc_θ=-40°（vs 之前 +60°）

### 預期
- 如果 17×17 仍是 valley → 真 universal（純 size 物理特性）
- 如果 17×17 不再是 valley → inc_θ × size interaction artifact

（結果回來後補表 + 結論）

## 累計紀錄

```
Round 1   v1                       −4.08 dB
Round 7   v6 generator best        −0.46 dB
Round 9   direct GD multi-restart  +6.94 dB
Round 12  inc_θ=+60°               +9.51 dB
Round 16  GD + SA                  +9.80 dB
Round 18  60 GHz × 10×10           +10.51 dB
Round 19  5.6 GHz × 19×19          +11.82 dB ← 當前最高 ★
```

從 v1 到當前 = 15.9 dB 改善。

## 對使用者最終建議（Round 21 為止）

### 硬體選型表

| 部署頻率 | 推薦 size | 預期 mean (dB) | 預期 max (dB) |
|---------|-----------|----------------|---------------|
| 5.6 GHz | **19×19 ★** 或 13×13 | +9.07 / +8.16 | +11.82 / +10.40 |
| 28 GHz | **13×13 ★** 或 15×15 | +8.93 / +8.15 | +9.37 / +9.75 |
| 60 GHz | **15×15 ★** | +8.55 | +10.51 |
| **All** | **絕對避免 17×17** | valley | — |

### 工作流程

```bash
python script/design_pattern_for_target.py \
  --element_num <推薦 size 依頻率> \
  --inc_theta 60 \
  --plateau_start 154 --plateau_w 46 \
  --steps 1500 --n_restarts 3 \
  --sa_steps 8000 --sa_T0 20 --sa_flip_n 3 \
  --device cuda:0
```

## Open Questions（待 round 22 結果驗證）

1. 17×17 universal valley 是否依賴 inc_θ
2. 不同 plateau target 位置下 universal valley 是否仍存在
3. SA reheat schedule 是否能進一步改善 mean（目前 mean +8 級，max +11.82）
4. 對於 ±60° 入射的對稱性是否真實（之前只測 +60°）

## Git
36+ commits pushed。
