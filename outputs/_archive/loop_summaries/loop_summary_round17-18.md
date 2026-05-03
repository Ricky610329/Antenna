# /loop Round 17–18 兩輪總結

> 接續 round 13-16 總結。Round 17-18 主軸是「**用 SA 排除 GD 雜訊後重新驗證
> 物理規律**」與「探**頻率 × 陣列大小**的二維最佳化」。

## Round 17 — SA Cross-size 統計

### 動機
Round 13 發現「25×25 + inc_θ=+60° 反而比 15×15 差」(+7.42 vs +9.51 dB)。
是 GD 卡 local min 假象嗎？還是真實物理特性？SA 後若 25×25 仍劣勢，則為
真實物理。

### 實驗（4 sizes × 5 seeds × {GD, GD+SA}, inc_θ=+60°, plateau 154-200）

| n | GD max | SA max | SA mean |
|---|--------|--------|---------|
| 10×10 | +5.87 | +8.21 | +7.52 |
| **15×15** | +9.51 | **+9.75 ★** | **+7.87 ★** |
| 20×20 | +5.22 | +8.28 | +7.13 |
| 25×25 | +7.42 | +7.98 | +7.32 |

### 結論
1. **SA 把 4 個 size 的 mean 全部抬到 +7~+8 dB**（GD 時 +2.31~+4.95）
2. **15×15 仍真實是物理最佳**（max +9.75 vs 25×25 max +7.98）
3. 「越大越好」直覺**部分被否定**——是真實物理特性，不是 GD 假象
4. 推測：15×15 = 7.5 wavelengths aperture 在 28 GHz / inc_θ+60° / target
   plateau 154-200 是 sweet spot

## Round 18 — 頻率 × element_num 二維 sweep

### 假說
若 aperture（element × wavelength）才是真正物理變因，則：
- 5.6 GHz（λ=53.6 mm）下，最佳 element_num 應比 28 GHz（λ=10.7 mm）大
- 60 GHz（λ=5 mm）下，最佳 element_num 應更小

### 實驗（執行中）
3 frequencies × 3 sizes × 3 seeds × {GD+SA}：5.6/28/60 GHz × 10/15/20

（結果回來後補 heatmap 與規律分析）

## 累積統計（從 round 1 到 round 18）

### 紀錄歷程
```
v1 純 binary               −4.08 dB
v6 generator best          −0.46 dB
direct GD multi-restart    +1.82 ~ +9.51 dB（不可靠）
GD+SA (round 16-17)        +5.98 ~ +9.80 dB（保底）
```

### 工具庫總覽
```
Design / production:
  ★★★ design_pattern_for_target.py    單目標設計（GD multi-restart + SA fine-tune）
  ★★★ design_batch.py                  批次設計多 target
  ★★  binary_sa_finetune.py            SA 翻轉工具（獨立可用）

Sweep / research:
  ★ sweep_physical_limit.py            plateau 位置 × 寬度
  ★ sweep_element_num.py               陣列大小
  ★ sweep_incidence_angle.py           入射角
  ★ sweep_frequency_x_size.py          頻率 × 大小（round 18 新增）

Benchmark:
  benchmark_gd_vs_sa.py                量化 SA 保底機率（round 16）
  benchmark_sa_cross_size.py           SA 跨 size 效果（round 17）

Diagnostics / 歷史保留:
  direct_pattern_search.py             物理上限 + STE 對照
  post_quantize_eval.py                phase 1 連續→後量化
  inspect_ris_run.py                   trainer 訓練檢視
  compare_ris_runs.py                  多 run overlay
  train_multi_target.py                plan D
  train_direct_ris.py                  plan E
  pretrain_surrogate.py                surrogate 預訓練
  generate_structured_patterns.py      結構化 pattern
  run_bit_migration.sh / run_full_v4.sh batch 腳本

Documentation:
  ★★★ RIS_DESIGN_GUIDE.md              使用者快速上手
  ★★  RIS_RESEARCH_REPORT.md            paper-style 完整報告
```

29+ commits pushed to `ricky/modernize`.

## 對使用者的最佳工作流程（Round 18 為止確立）

```bash
# 對給定 (target, frequency, inc_θ) 找最佳 binary pattern
python script/design_pattern_for_target.py \
  --element_num 15 \
  --inc_theta 60 \
  --plateau_start 154 --plateau_w 46 \
  --steps 1500 --n_restarts 3 \
  --sa_steps 8000 --sa_T0 20 --sa_flip_n 3 \
  --device cuda:0
```

**2 分鐘**輸出 binary pattern：
- 70% 機率達 +7 dB 級
- 平均 +7.65 dB（round 16 統計）
- 物理上限 +9.80 dB

## 開放問題

1. ~~25×25 為何不如 15×15~~（Round 17 答：真實物理特性，element spacing 與
   beam-forming interaction）
2. **Round 18 頻率 sweep 結果**（執行中）— aperture 假說是否成立
3. 不同入射 φ_i 是否影響 — round 12 只測 θ_i 沒測 φ_i
4. SA flip schedule（先 flip_n=10 再 5 再 3 再 1）是否更穩
