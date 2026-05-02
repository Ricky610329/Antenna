# /loop Round 31–32 兩輪總結

> 接續 round 29-30 總結。Round 31 修了 design tool critical API bug，Round 32
> 確認 +11.82 dB 是 highly target-specific（broadside only）。

## Round 31 — Reproducibility 真相 + API 修正

### CUDA Determinism Test
3 modes × 3 runs each, seed=0, 5.6 GHz × 19×19 × inc_θ=+60°:
- Mode A 預設: +11.8231 / +11.8231 / +11.8231
- Mode B cudnn.deterministic: +11.8231 / +11.8231 / +11.8231
- Mode C use_deterministic_algorithms: +11.8231 / +11.8231 / +11.8231

**所有 9 次 byte-identical**——GD 完全 deterministic！

### Critical API Bug Found
Round 30「seed=0 不一致」是 design_pattern_for_target.py **缺 `--freq` 參數**！
Round 30 命令用 `--inc_theta 60`，但沒有方法指定 freq → default 28 GHz。
+4.78 是 28 GHz × 19×19 × +60° 的合理結果，不是 5.6 GHz 配置。

### 修正
- `design_pattern_for_target.py` 加 --freq
- `design_batch.py` 加 --freq
- 加 RISSimulator 配置 logging

### 驗證
```bash
python script/design_pattern_for_target.py --freq 5.6e9 --element_num 19 \
  --inc_theta 60 --plateau_start 154 --plateau_w 46 --seed 0 ...
→ restart 1/1: suppression=+11.82 dB ★ 立刻重現
```

## Round 32 — +11.82 是 Target-Specific

### 實驗
5 plateau positions × 5.6 GHz × 19×19 × inc_θ=+60° × seed=0：

| name | θ_center | suppression |
|------|----------|-------------|
| **center** | **-1.5°（broadside）** | **+11.82 ★** |
| right | +30° | +6.68 |
| far_left | -48.5° | +4.70 |
| left | -33° | +1.99 |
| far_right | +61.5° | +0.60 ↓ |

mean **+5.16**, max **+11.82**

### 重大發現
1. **+11.82 dB 只在 broadside (θ_center≈0°) 達到**——highly target-specific
2. **其他方向 target 只 +0.6~+6.7 dB**——遠低於 broadside
3. **物理**：inc_θ=+60° → specular reflection 在 -60°；broadside 最遠離 specular，
   最容易做 directional shaping；其他方向受 specular 干擾

### 修正使用者建議
之前廣義說「5.6 GHz × 19×19 × inc_θ=+60° 是最佳硬體配置」**只對 broadside target 成立**。

實務工作流程（target-aware 配置選擇）：
- **broadside target (θ≈0°)**: 5.6 GHz × 19×19 × inc_θ=+60° → +11.82 dB
- **其他方向 target**: 各別跑 fine-grid sweep 找最佳 (freq, n, inc_θ) 配置

## 累計圖譜（32 rounds）

### 工具庫（53+ commits）
```
Design / production:
  ★★★ design_pattern_for_target.py    (with --freq, multi-restart, SA reheat=2)
  ★★★ design_batch.py                  (with --freq, CSV/CLI)
  ★★  binary_sa_finetune.py            (含 reheat 模式)

Sweep:
  sweep_physical_limit.py
  sweep_element_num.py
  sweep_incidence_angle.py
  sweep_frequency_x_size.py
  sweep_inc_x_size_2d.py
  sweep_inc_phi.py

Benchmark:
  benchmark_gd_vs_sa.py (with --freq)
  benchmark_sa_cross_size.py
  benchmark_sa_reheat.py
  benchmark_sa_aggressive.py
  test_determinism.py (round 31 新)

Diagnostics:
  direct_pattern_search.py
  post_quantize_eval.py
  inspect_ris_run.py / compare_ris_runs.py
  generate_structured_patterns.py
  ...

Documentation:
  ★★★ RIS_DESIGN_GUIDE.md
  ★★  RIS_RESEARCH_REPORT.md
  + 17 個 round summaries
```

### 紀錄歷程
```
v1                              −4.08 dB
v6 generator best               −0.46 dB
direct GD multi-restart         +1.82 ~ +9.51 dB
GD+SA reheat=2 (R25)            mean +8.38, max +9.75
GD+SA 5.6 GHz × 19 broadside    +11.82 dB ← target-specific 物理上限
```

從 v1 到 reproducible best (broadside) = **15.9 dB 改善**

## Open Questions

1. ~~+11.82 是運氣命中~~（R31 確認 reproducible）
2. ~~+11.82 是 universal 最佳~~（R32 確認 broadside-specific）
3. **每個 target 方向最佳 (freq, n, inc_θ) 配置是什麼？**——需要更大 sweep
4. GPU-batched SA 加速（剩餘 engineering 改善）
5. Array factor 數學是否能預測 broadside dominance
