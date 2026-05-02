# /loop Round 33–34 兩輪總結

> 接續 round 31-32 總結。Round 33-34 主軸：用 target × inc_θ sweep 探究
> 是否有 specular-avoidance 物理規律可預測最佳硬體配置。

## Round 33 — Specular-Avoidance 假說失敗

### 動機
Round 32 發現 +11.82 是 broadside-specific。物理推測：specular reflection 在
-inc_θ；target 越遠離 specular 越容易做 directional shaping。預期 anti-diagonal
heatmap 結構（target 越正，inc 越負）。

### 實驗（5 target × 5 inc × 5.6 GHz × 19×19, GD only seed=0）

| target | best inc_θ | suppression | 預期最佳 inc | 符合假說 |
|--------|-----------|-------------|--------------|---------|
| -50° | -60° | +5.86 | +60° (specular -60°) | ✗ |
| -25° | -30° | +6.69 | +60° | ✗ |
| **0°** | **+60°** | **+8.79 ★** | -60° | ✗ |
| +25° | -60° | +6.08 | -60° | ✓ (但 -30° 應更好) |
| +50° | +60° | +6.40 | -60° | ✗ |

### 結論
**Specular-avoidance 假說失敗**：target × inc 配對 chaotic，無單一物理公式。
grating lobe + element pattern + 干涉的多因子 interaction 太複雜。

## Round 34 — 25 Cells with SA Fine-tune

### 動機
Round 33 是 GD only 結果，加 SA 是否能把所有 cells 拉到 +6 級？dead spots
（如 target=+25°×inc=+30° = +1.71）能否被 SA 救？

### 實驗（執行中）
5 target × 5 inc × 5.6 GHz × 19×19 × GD+SA reheat=2, 25 cells, ~12 min on GPU。

### 預期
- 若全部 cells SA 後 ≥+6 dB → SA 是「萬能 saver」，dead spots 是 GD 假象
- 若仍有 dead spots → 真實物理 dead zones 存在

## 對使用者最終工作流程（roundtrip）

```bash
# Step 1: 對自己 target 跑 inc_θ sweep 找最佳
# (大致 inc=±60° 都好)

# Step 2: 用 design tool 拿 binary pattern
python script/design_pattern_for_target.py \
  --element_num 19 --freq 5.6e9 --inc_theta 60 \
  --plateau_start <根據 target> --plateau_w 46 \
  --steps 1500 --n_restarts 5 \
  --sa_steps 8000 --sa_reheat_cycles 2 \
  --device cuda:0
```

## 累計（34 rounds）
- 紀錄歷程：v1 −4.08 → broadside +11.82 dB
- **55+ commits**, **18 個 round summaries**
- **16 個 scripts**（design / sweep / benchmark / diagnostics）
- **3 層完整文檔**

## Open Questions

1. ~~Specular-avoidance 假說~~（R33 否定）
2. **SA 是否能救所有 dead spots**（R34 探，執行中）
3. GPU-batched SA 加速
4. Array factor 數學能否預測 chaotic patterns
