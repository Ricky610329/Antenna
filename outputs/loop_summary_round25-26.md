# /loop Round 25–26 兩輪總結

> 接續 round 23-24 總結。Round 25 完成 SA reheat schedule 改善。Round 26
> 補完 inc_phi 入射方位角影響圖譜。

## Round 25 — SA Reheat=2 工程化最佳

### 動機
Round 16 SA single-cycle mean +7.65 dB，仍離 ceiling +9.80 dB 有距離。
Reheat schedule（多輪 cooling，每輪從 best 重啟）能否突破？

### 實驗（5 seeds × 3 reheat configs）

| Config | mean | std | min | max |
|--------|------|-----|-----|-----|
| GD-only | +4.95 | 2.56 | — | — |
| reheat=1 | +7.87 | 1.29 | +5.98 | +9.75 |
| **reheat=2** ★ | **+8.38** | **0.85** | **+7.13** | +9.75 |
| reheat=4 | +8.34 | 1.04 | +7.16 | +9.75 |

### 結論
1. **reheat=2 比 single-cycle 高 +0.51 dB mean**
2. **std 縮 34%** (1.29 → 0.85) — 結果更可預測
3. **worst case +1.15 dB** (+5.98 → +7.13) — reliability 提升
4. **reheat=4 沒額外好處**（diminishing returns）
5. **Seed 0 戲劇性案例**：reheat=1 +5.98 → reheat=2 +8.36（+2.38 dB 大躍升）

整合：design_pattern_for_target.py / design_batch.py 預設 sa_reheat_cycles=2。

## Round 26 — inc_phi 入射方位角 sweep

### 動機
之前所有 sweep（round 9-25）都固定 inc_phi=90°（default）。完整圖譜需要補 phi 維度。

### 實驗（執行中）
5 phi × 5 sizes × 2 seeds × GD+SA reheat=2 = 50 designs
- inc_phi: 0 / 45 / 90 / 135 / 180°
- inc_theta=+60°（之前確認 best）
- 28 GHz, plateau 154-200

（結果回來後補 heatmap）

### 預期
- 若 phi 影響顯著 → 加入使用者選型表
- 若 phi 不顯著 → 確認 phi=90° 為 robust default

## 紀錄歷程

```
v1                          −4.08 dB
v6 generator best           −0.46 dB
GD multi-restart            +1.82 ~ +9.51 dB
GD+SA single-cycle (R16)    +5.98 ~ +9.80 dB (mean +7.65)
GD+SA reheat=2 (R25)        +7.13 ~ +9.75 dB (mean +8.38) ← 工程化最佳
GD+SA 5.6 GHz × 19×19 (R19) +11.82 dB ← 物理紀錄
```

## 工具庫累計（30+ scripts, 42+ commits）

```
Design / production:
  ★★★ design_pattern_for_target.py    (GD multi-restart + SA reheat=2)
  ★★★ design_batch.py
  ★★  binary_sa_finetune.py            (含 reheat 模式)

Sweep / research:
  sweep_physical_limit.py            plateau pos × width
  sweep_element_num.py               陣列大小
  sweep_incidence_angle.py           入射 theta
  sweep_frequency_x_size.py          頻率 × 大小
  sweep_inc_x_size_2d.py             2D heatmap inc_θ × size
  sweep_inc_phi.py                   入射 phi（round 26 新）

Benchmark:
  benchmark_gd_vs_sa.py              SA 保底機率
  benchmark_sa_cross_size.py         SA 跨 size 效果
  benchmark_sa_reheat.py             reheat schedule（round 25 新）

Diagnostics / 歷史:
  direct_pattern_search.py
  post_quantize_eval.py
  inspect_ris_run.py
  compare_ris_runs.py
  train_multi_target.py / train_direct_ris.py
  pretrain_surrogate.py
  generate_structured_patterns.py
  run_bit_migration.sh / run_full_v4.sh

Documentation:
  ★★★ RIS_DESIGN_GUIDE.md
  ★★  RIS_RESEARCH_REPORT.md
  + 12 個 round summaries
```

## 對使用者最終工作流程（Round 25 確立）

```bash
python script/design_pattern_for_target.py \
  --element_num 15 \
  --inc_theta 60 \
  --plateau_start 154 --plateau_w 46 \
  --steps 1500 --n_restarts 3 \
  --sa_steps 8000 --sa_T0 20 --sa_flip_n 3 \
  --sa_reheat_cycles 2 \
  --device cuda:0
```

**2 分鐘** → mean +8.38 dB / max +9.75 dB / worst case +7.13 dB suppression
（vs generator-based v6 -0.46 → +10 dB 改善）

## Open Questions

1. ~~Universal valley~~（R22 否定，R24 確認 chaotic）
2. **inc_phi 影響**（R26 探，執行中）
3. 不同 plateau 位置下整體規律
4. SA + continuous GD on logits interleaved 是否能突破 ceiling
5. 物理理論：array factor 數學能否預測 chaotic valley
