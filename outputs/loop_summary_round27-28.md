# /loop Round 27–28 兩輪總結

> 接續 round 25-26 總結。Round 27 完成 inc_phi 入射方位角 sweep（補完 RIS
> 物理特性圖譜）。Round 28 驗證 +11.82 dB 紀錄是否可重現。

## Round 27 — inc_phi 影響非常大

### 實驗（5 phi × 5 sizes × 2 seeds × GD+SA reheat=2, inc_θ=+60°, 28 GHz）

| phi | row mean | 評論 |
|-----|----------|------|
| **0°** | +4.76 | ↓↓ **絕對避免** |
| +45° | +7.86 | OK |
| **+90° (default)** | **+8.30 ★** | robust optimal |
| +135° | +8.36 | OK |
| **180°** | +6.09 | ↓ **避免** |

### 重大發現
1. **phi=0/180° 災難**——比 ±90° 低 ~3-4 dB
2. **phi=±45/90/135° 都好**——範圍 +7.86 ~ +8.36 dB
3. **物理解讀**：response 是 phi=0° 切片；inc_phi=0/180° 入射與 response 同
   平面 → specular reflection 干擾 sidelobe 區。phi=90° 入射與 response 正交
   → 反射波遠離 sidelobe → 高 suppression
4. **RISSimulator default phi=90° 是 robust optimal**

## Round 28 — 驗證 +11.82 dB 紀錄

### 動機
Round 19 在 5.6 GHz × 19×19 找到 +11.82 dB single-seed max。是 outlier 還是
真實上限？跑 10 seeds × GD+SA reheat=2 看可重現性。

### 實驗（執行中）
（結果回來後補表）

## 累計重要圖譜（Rounds 1-27 統合）

### 物理可達區圖譜
- **inc_θ**：偏角 ±60° 略好（mean 差距 ~1 dB SA 後）
- **inc_phi**：必須避免 0/180°（差 3-4 dB）；±45/90/135° 都 OK
- **element_num**：依配置 chaotic（無公式可推），需 fine grid sweep
- **frequency**：5.6 GHz × 19×19 達 +11.82（單 seed），28 GHz × 13×13 +9.37 max
- **plateau width**：寬比窄略好（round 9 finding）

### 工具完整度（44+ commits）
```
Design / production:
  ★★★ design_pattern_for_target.py    (GD multi-restart + SA reheat=2)
  ★★★ design_batch.py                  (CSV/CLI batch)
  ★★  binary_sa_finetune.py            (含 reheat 模式)

Sweep:
  sweep_physical_limit.py              (plateau pos × width)
  sweep_element_num.py                 (陣列大小)
  sweep_incidence_angle.py             (inc_θ)
  sweep_frequency_x_size.py            (頻率 × 大小)
  sweep_inc_x_size_2d.py               (2D heatmap)
  sweep_inc_phi.py                     (inc_phi, round 27)

Benchmark:
  benchmark_gd_vs_sa.py                (SA 保底機率)
  benchmark_sa_cross_size.py           (SA 跨 size 效果)
  benchmark_sa_reheat.py               (reheat schedule)

Diagnostics:
  direct_pattern_search.py
  post_quantize_eval.py
  inspect_ris_run.py / compare_ris_runs.py
  train_multi_target.py / train_direct_ris.py
  pretrain_surrogate.py
  generate_structured_patterns.py
  run_bit_migration.sh / run_full_v4.sh

Documentation:
  ★★★ RIS_DESIGN_GUIDE.md
  ★★  RIS_RESEARCH_REPORT.md
  + 14 個 round summaries
```

## 紀錄歷程

```
v1                          −4.08 dB
v6 generator best           −0.46 dB
GD multi-restart            +1.82 ~ +9.51 dB
GD+SA single-cycle (R16)    mean +7.65, max +9.80
GD+SA reheat=2 (R25)        mean +8.38, max +9.75 ← 工程化最佳
GD+SA phi=90° (R27)         mean +8.30 (consistent)
GD+SA 5.6 GHz × 19×19 (R19) +11.82 dB ← 物理紀錄（單 seed，待 R28 驗證）
```

## Open Questions

1. **+11.82 dB 紀錄可重現性**（R28 驗證中）
2. 不同 plateau 位置下整體規律是否 robust
3. SA + continuous GD interleaved 是否突破 +9.80 ceiling
4. Array factor 數學能否預測 chaotic valley
