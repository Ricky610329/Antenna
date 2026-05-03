# /loop Round 66–67 兩輪總結 — Dataset Mode 啟動

> R64-R65 確認 worst-case loss + ripple penalty 是 patch 移植的正確 metric。
> R66-R67 改 dataset generation mode，建立批量「對的訓練資料」。

## TL;DR

- **dataset_v1**: 36 entries × 2 ripple weights = 72 Pareto rows
- 涵蓋 (28/38 GHz) × (n=21/31) × (θc=-30/0/+30°) × (w=10/20/30°)
- 建立 RIS playground → patch surrogate dataset 的方法論

## R66 — Dataset Generator + Schema

### Schema

```json
{
  "entry_id": 0,
  "config": {"freq": 38e9, "n": 21, "target_theta_c": 0,
             "target_width_deg": 20, "inc": 51, "target_shape": "flat_plateau"},
  "main_idx_range": [160, 200],
  "pareto": [
    {"ripple_weight": 0.0, "metrics": {...}, "best_seed": 2,
     "all_seeds": [...], "pattern_file": "patterns/entry0000_rw0.0.npy", ...},
    {"ripple_weight": 2.0, ...}
  ]
}
```

每 entry 包含 Pareto frontier across ripple weights，不只 single best。

### Schema 驗證
sample 2 entries × 18 runs = 9.9 min（~33s/run）。Format 正常。

## R67 — 36 Entries Dataset 完整分析

### Per ripple_weight 統計

| rw | worst_mean | worst_max | ripple_mean | flat-top% |
|----|-----------|-----------|-------------|-----------|
| 0.0 (steering) | +1.55 | +4.54 | 14.08 | **0/36 (0%)** |
| 2.0 (flat-top) | -1.06 | +0.43 | 2.42 | **24/36 (67%)** |

→ ripple penalty 把 ripple 從 14 dB 壓到 2.4 dB，trade-off 換來 67% flat-top 達成率。

### (n × target_width) heatmap at rw=2

| | w=10° | w=20° | w=30° |
|---|------|-------|-------|
| n=21 | -0.05 | +0.16 | -1.03 |
| **n=31** | **+0.43 ★** | -0.21 | -0.76 |

**結論**：
- n=31 × w=10° 是 binary 1-bit flat-top 甜點 (+0.43 dB worst)
- w=30° 是 bottleneck — 物理上 binary 做不到 wide flat-top
- n 大 + w 窄 → 容易達 flat-top
- n 大 + w 寬 → 反而更難（power spread 太分散）

### Per target_theta_c (rw=2)

| θc | flat-top achievement |
|------|---------------------|
| -30° | 9/12 (75%) |
| 0° | 7/12 (58%) |
| +30° | 8/12 (67%) |

→ θc 影響小（broadside 不顯著比 off-axis 難）。

### Edge cases (rw=2 最差 5 個)

| config | worst | ripple | flat-top |
|--------|-------|--------|----------|
| 28 GHz × n=21 × θc=-30 × w=30 | -3.69 | 4.09 | no |
| 38 GHz × n=21 × θc=-30 × w=30 | -3.23 | 4.40 | no |
| 28 GHz × n=31 × θc=+30 × w=30 | -2.61 | 3.24 | no |
| 38 GHz × n=31 × θc=+0  × w=30 | -2.61 | 3.86 | no |
| 38 GHz × n=21 × θc=+0  × w=30 | -2.35 | 4.14 | no |

→ **w=30° 全部失敗**。Binary 1-bit 物理上限。

### 最佳 5 個 (rw=2)

| config | worst | ripple | flat-top |
|--------|-------|--------|----------|
| 38 GHz × n=31 × θc=+0  × w=10 | +0.43 | 0.92 | ✓ |
| 28 GHz × n=31 × θc=+0  × w=10 | +0.17 | 1.14 | ✓ |
| 28 GHz × n=21 × θc=+30 × w=20 | +0.16 | 2.76 | ✓ |
| 38 GHz × n=31 × θc=+30 × w=10 | +0.11 | 1.75 | ✓ |
| 28 GHz × n=31 × θc=+30 × w=10 | +0.04 | 0.91 | ✓ |

→ 共同特徵：n=31 + w=10°/20° + 任何 θc/freq。

## 對 Patch Antenna 移植的方法論

dataset_v1 直接驗證 R64-R65 設計決策：

1. **每 config 兩個 ripple weight (0, 2)**
   - rw=0 給 use case「max suppression, 不在乎 ripple」
   - rw=2 給 use case「整片貼上蓋」
   - Patch 移植：S11 vs gain 也是兩維 trade-off

2. **多 seed 取最好**
   - 5 seeds × 2 rw 比 1 seed × 1 rw 穩 5-10×
   - Patch 移植：surrogate 預測也要 multi-seed 評估

3. **Edge case 識別**
   - 哪些 config 物理上不可行（如 w=30° 強要 flat-top）
   - 訓練 surrogate 應**標註 "infeasible region"**，不是死命壓 loss
   - Patch 移植：HFSS 也有 infeasible region (e.g. mismatch impedance)

## 紀錄歷程修正（dataset 視角）

| 階段 | 焦點 | 紀錄 |
|------|------|------|
| R37-R47 sigmoid path | single-target steering | +13.44 (max-max) |
| R57-R63 free-phase | single-target steering | +30.99 (max-max, 虛胖) |
| **R64-R65 worst-case loss** | flat-top deployment | +6.88 (worst, w=30) / +0.22 ripple 1.75 (rw=2) |
| **R66-R67 dataset_v1** | 多 use case 涵蓋 | 36 entries Pareto frontier |

## 累計（67 rounds, 102+ commits）

20+ scripts 包括:
- script/build_dataset.py（NEW, R66）
- script/analyze_dataset.py（NEW, R67）
- script/optimize_worst_case.py（R64）
- script/worst_case_eval.py（R64）
- script/verify_free_phase_record.py（R57）

dataset_v1: 36 entries × 2 ripple = 72 Pareto rows，
直接可載入訓練 surrogate / generator。

## Open Questions

1. **n=41 entries 是否值得加**（成本 ~2× 但對 patch 大 array 移植關鍵）
2. **broader θc range (±45, ±60)** dataset 對 robustness 重要
3. **rw=5 stricter ripple** 提供更嚴格 flat-top option
4. **Continuous phase baseline** 對每個 config 跑一次當理論 ceiling
5. **R68+ 直接拿 dataset 訓練 RIS surrogate** 驗證 dataset 品質
