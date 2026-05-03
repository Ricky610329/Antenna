# /loop Round 23–24 兩輪總結

> 接續 round 21-22 總結。Round 21 提了「universal valley at 17×17」假說，
> Round 22 用 inc_θ=-40° 驗證後**否定**。Round 23-24 繼續探究 valley 移動規律。

## Round 23 — inc_θ × element_num 二維 sweep

### 動機
Round 22 揭露 valley 隨 inc_θ 改變位置（+60° valley=17, -40° valley=13）。
若有規律，可能有 grating lobe 數學公式可推導；若散亂，必須每個配置 case-by-case。

### 實驗（執行中）
7 inc_θ ({-60, -40, -20, 0, +20, +40, +60}) × 5 sizes ({11, 13, 15, 17, 19})
× 2 seeds × {GD+SA}, 28 GHz, plateau 154-200

35 designs total，~25 min on GPU。

### 部分結果（前 18/35）

| inc_θ | 11 | 13 | 15 | 17 | 19 |
|-------|-----|-----|-----|-----|-----|
| -60° | ? | ? | ? | ? | ? |
| -40° | ? | ? | ? | +8.28 | +7.08 |
| -20° | +7.28 | +8.05 | +8.15 | +7.46 ↓ | +7.78 |
| 0° | +7.33 | +7.88 | +7.77 | ? | ? |

初步觀察：
- inc_θ=-40° 在 17×17 達 +8.28（**peak**），跟之前 round 22 數據一致（17 不是 valley）
- inc_θ=-20° 17×17 是 valley（+7.46）
- valley 在 inc_θ ∈ [-40°, -20°] 之間移動

（完整結果回來後補上 heatmap）

## Round 24 — 待完整 2D heatmap 後分析

### 計畫
1. 完整 7×5 heatmap 看 valley 移動軌跡
2. 若沿對角線 → 推導物理公式
3. 若散亂 → 確認 case-by-case sweep 是必要工具

## 紀錄歷程

```
v1                          −4.08 dB
v6 generator best           −0.46 dB
direct GD multi-restart     +1.82 ~ +9.51 dB
GD+SA 28 GHz                +9.80 dB
GD+SA 60 GHz × 10×10        +10.51 dB
GD+SA 5.6 GHz × 19×19       +11.82 dB ← 當前最高 ★
```

## 工具庫累計

```
Design / production:
  ★★★ design_pattern_for_target.py  GD multi-restart + SA fine-tune
  ★★★ design_batch.py                批次設計多 target
  ★★  binary_sa_finetune.py          SA 翻轉工具

Sweep / research:
  ★ sweep_physical_limit.py            plateau 位置 × 寬度
  ★ sweep_element_num.py               陣列大小
  ★ sweep_incidence_angle.py           入射角
  ★ sweep_frequency_x_size.py          頻率 × 大小
  ★ sweep_inc_x_size_2d.py             inc_θ × 大小（round 23 新）

Benchmark:
  benchmark_gd_vs_sa.py                量化 SA 保底機率
  benchmark_sa_cross_size.py           SA 跨 size 效果

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
  + 11 個 round summaries
```

38+ commits pushed.

## Open questions（持續累積）

1. ~~Universal valley at 17×17~~（Round 22 否定）
2. **Valley 在 (inc_θ, size) 空間的軌跡**（Round 23-24 探）
3. 是否有 grating lobe 公式可預測 valley
4. 不同 target plateau 位置下規律是否相同
5. SA reheat schedule 是否進一步改善 mean
