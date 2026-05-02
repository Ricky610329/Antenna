# /loop Round 41–42 兩輪總結

> 接續 round 39-40 總結。Round 41 確認 5.6 GHz × 19×19 (9.5λ) 是 broadside
> sweet aperture。Round 42 細網格探 inc 是否 +60° 真最佳。

## Round 41 — Sweet Aperture 確認（不是越大越好）

5.6 GHz × broadside × inc=+60° × SA-per-restart × 5 restart：

| size | aperture | suppression |
|------|----------|-------------|
| 11×11 | 5.5λ | +8.92 |
| 13×13 | 6.5λ | +9.75 |
| 15×15 | 7.5λ | +9.22 |
| 17×17 | 8.5λ | +9.66 |
| **19×19** | **9.5λ** | **+11.82 ★** |
| 21×21 | 10.5λ | +8.59 ↓ |

### 重要發現
1. **19×19 (9.5λ) 仍是物理最佳**——新邏輯確認
2. **21×21 (10.5λ) 反而下降**——aperture 越大不好
3. **9.5λ 是 sweet aperture**——超過某 size 後劣化

## Round 42 — Inc 細網格探（執行中）

### 動機
之前 inc sweep 用 5 個值：-60, -30, 0, +30, +60。+60° 是最佳，但是否 +55, +65
有更好？細網格驗證。

### 實驗
5.6 GHz × 19 × broadside × inc ∈ {+50°, +55°, +65°, +70°} × 5 restart × SA-per-restart
（vs +60° baseline = +11.82 dB）

（結果回來後補表）

## 跨頻率 Sweet Aperture 對比

| Frequency | Sweet aperture | Sweet n | Max suppression |
|-----------|----------------|---------|-----------------|
| 5.6 GHz | **9.5λ** | 19×19 | **+11.82 ★** |
| 28 GHz | 6.5λ | 13×13 | +10.11 (+45.5° best) |
| 60 GHz | 7.5λ | 15×15 | +9.91 |

不是線性 scaling — 不同頻率有自己的 sweet aperture。

## Epistemic 進展（截至 R41）

| Round | 假說 | 結果 |
|-------|------|------|
| R12 | ±60° inc 最佳 | R34 部分否定 → ±30° 在某些 target |
| R32 | +11.82 universal | broadside-only |
| R33 | specular-avoidance | chaotic |
| R35 | best GD → best SA | 修為 SA-per-restart |
| R36 | 新邏輯破 +11.82 | 仍 +11.82 |
| R37 | mean +5.16 是真實 | 否定 → +9.67 with new logic |
| R39/40 | aperture 主導 best target | 確認（aperture ≥7.5λ broadside-best, 6.5λ 偏側）|
| **R41** | **越大越好 aperture** | **否定（9.5λ 是 sweet, 10.5λ 下降）** |

## 累計（42 rounds, 68+ commits）

### 工具庫
17+ scripts:
- 3 design tools (with SA-per-restart, --freq, --inc_theta)
- 6 sweep tools
- 4 benchmark tools (含 test_determinism)
- 多個 diagnostics
- 3 層完整文檔 + 25 round summaries

### 紀錄歷程
```
v1                              −4.08 dB
v6 generator best               −0.46 dB
GD+SA reheat=2 (R25)            +9.75 dB
SA-per-restart 5.6 GHz × 19 (R37) mean +9.67, max +11.82
+60° broadside (R19)            +11.82 dB ★ 物理上限保持（42 rounds 確認）
```

從 v1 到當前 mean 最佳 = **13.75 dB** 改善

## 對使用者最終建議（R41 為止）

### 硬體選型表
| 部署頻率 | 推薦 size | aperture | best target | mean |
|---------|-----------|----------|-------------|------|
| **5.6 GHz** | **19×19** | 9.5λ | broadside | **+9.67 ★** |
| 28 GHz | 13×13 | 6.5λ | +45.5° | +8.91 |
| 60 GHz | 15×15 | 7.5λ | broadside | +9.10 |

### 工作流程（最佳 setup）
```bash
python script/design_pattern_for_target.py \
  --element_num 19 \
  --freq 5.6e9 \
  --inc_theta 60 \
  --plateau_start 154 --plateau_w 46 \
  --steps 1500 --n_restarts 5 \
  --sa_steps 8000 --sa_reheat_cycles 2 \
  --device cuda:0
```

5-7 分鐘 → expect mean +9 dB, max +11.82 dB (broadside)，worst ~+8 dB。

## Open Questions

1. **Inc fine grid 是否有超 +60° 更好的角度**（R42 探）
2. 60 GHz × 15 是否能用更激進 SA 突破 +9.91
3. GPU-batched SA 加速
