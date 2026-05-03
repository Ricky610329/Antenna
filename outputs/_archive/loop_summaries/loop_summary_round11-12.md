# /loop Round 11–12 兩輪總結

> 接續 round 9-10 總結。Direct GD path 已完整建立，這兩輪聚焦**對使用者實用性
> 的工具改善與物理特性研究**。

## Round 11 — Batch design 工具

### 工具
`script/design_batch.py`：對 N 個 target 批次設計 binary RIS pattern。

### 兩種輸入：

**CSV**：
```csv
name,plateau_start,plateau_w,n_restarts
north,140,46,5
east,217,46,5
south,250,40,5
```
```bash
python script/design_batch.py --csv targets.csv
```

**CLI list**：
```bash
python script/design_batch.py \
  --target north:140:46 \
  --target east:217:46 \
  --target south:250:40 \
  --n_restarts 5
```

### Demo 結果（25×25 RIS, 5 targets, 3 restarts）

| name | θ_center | suppression |
|------|----------|-------------|
| north | -8.5° | +7.37 |
| east | +30.0° | +8.65 ← **新紀錄** |
| south | -48.5° | +8.30 |
| west | +61.5° | +5.59 |
| wide_center | +2.0° | +6.38 |

mean **+7.26 dB**, max **+8.65 dB**

## Round 12 — 入射角 sweep + 整合總結

### 工具
`script/sweep_incidence_angle.py`：探不同 inc_θ (-60° ~ +60°) 對可達
suppression 的影響。

### 結果（15×15 RIS, 7 angles × 3 targets × 3 restarts）

| inc_θ | left | center | right | 平均 |
|-------|------|--------|-------|------|
| -60° | +6.84 | +5.88 | +6.20 | **+6.31** |
| -40° (default) | +5.32 | +3.24 | +7.04 | +5.20 |
| -20° | +4.07 | +5.33 | +4.34 | +4.58 |
| **0°（垂直）** | +1.01 | +2.78 | +2.01 | **+1.93 最差** |
| +20° | +4.75 | +5.00 | +5.57 | +5.11 |
| +40° | +8.45 | +4.75 | +4.53 | +5.91 |
| **+60°** | +5.00 | **+9.51** | +5.96 | **+6.82 最佳** |

### 關鍵物理發現
1. **垂直入射（θ_i=0°）是最差的安裝角** — 平均 +1.93 dB，僅為 ±60° 的 1/3
2. **偏角入射有顯著優勢** — ±60° 平均 +6.5 dB（比 default -40° 高 +1.3 dB）
3. **新紀錄 +9.51 dB**（inc_θ=+60°, target plateau 154-200, 15×15, 3 restarts）
4. **物理直覺驗證**：specular reflection 越偏向 grazing，可達 main beam 引導
   方向越廣，越容易隔離出純 sidelobe 區域

## 累積對所有 sweep 的物理結論

| 變因 | 觀察 | 對使用者建議 |
|------|------|--------------|
| RIS 邊長 | 25×25 (+7.57) >> 15×15 (+4.61) | 硬體選 25×25 |
| Plateau 寬度 | 寬 plateau 比窄高 1-2 dB | 設計時放寬 main beam region |
| Plateau 位置 | 邊緣 (±60°) > 中央 (0°) | 接近 specular 反射方向有優勢 |
| Multi-restart | 5 次取最佳比 1 次高 +0.65~+2 dB | 必開 |
| Inc_θ | 待 round 12 sweep | TBD |

## 工具庫總整理（21 個 commits 累計）

```
script/
├── design_pattern_for_target.py   ★★★ 為單一 target 設計（with multi-restart）
├── design_batch.py                ★★★ 批次設計 N 個 target（CSV/CLI）
├── sweep_physical_limit.py        ★★ 多 target heatmap
├── sweep_element_num.py           ★★ RIS 邊長對 suppression 影響
├── sweep_incidence_angle.py       ★ 入射角 sweep（round 12 新增）
├── direct_pattern_search.py       diagnostics — 物理上限 + STE 對照
├── post_quantize_eval.py          phase 1 連續 → 後量化評估
├── inspect_ris_run.py             trainer 訓練結果完整檢視
├── compare_ris_runs.py            多 run 跨 overlay
├── train_multi_target.py          plan D（multi-target generator + cond_reg）
├── train_direct_ris.py            plan E（無 surrogate、直接 RIS sim）
├── pretrain_surrogate.py          surrogate 預訓練（含 --n_structured）
├── generate_structured_patterns.py 線性相位梯度結構化 pattern
├── run_bit_migration.sh           一鍵 phase1+phase2+inspect
├── run_full_v4.sh                 一鍵 pretrain+phase1+phase2+inspect
└── RIS_DESIGN_GUIDE.md            ★★★ 使用者快速上手指引
```

## 下一步（round 13+）

1. 完成 round 12 入射角 sweep 結果分析
2. 不同頻率對比（5.6 / 28 / 60 GHz）— 改 RISSimulator 參數即可
3. 探 quantize bit 數（1-bit vs 2-bit）對 suppression 的影響
4. 整合所有 sweep 結果寫成「物理特性 paper-style summary」（如果是研究用途）
5. 收尾：把 `RIS_DESIGN_GUIDE.md` 整合到主 README
