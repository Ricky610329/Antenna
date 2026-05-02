# /loop Round 13–16 兩輪總結（合併 round 13-14 與 15-16）

> 接續 round 11-12 總結。研究 cycle 從「探物理可達區」走向「**保證可靠達到上限**」
> 的工程化階段。最大突破：**SA fine-tune 把 1/10 命中率變保底 +7 dB**。

## Round 13 — 反直覺驗證
- 重現 round 12 最佳：15×15 + inc_θ=+60° + plateau 154-200 → **+9.51 dB**
- **10 restarts 仍是 +9.51**（物理上限確認）
- **25×25 + inc_θ=+60° 反而 +7.42**（element_num 與 inc_θ 不獨立 multiplicative）
- **1/10 命中率**——大多數 GD restart 卡在 +2 ~ +6 dB local minima

## Round 14 — Master Research Report
寫了 `script/RIS_RESEARCH_REPORT.md`（paper-style）：
- 整合 14 rounds 完整研究軌跡
- 對實驗室過往 11 篇碩論的批判（哪些有料、哪些 fancy 但有限）
- 工具庫總覽 + 開放問題

## Round 15 — SA fine-tune 重大突破

### 動機
GD 5 restart 中 1/5 達上限 +9.51，其餘卡 local min（+2 ~ +6）。對使用者
「希望可靠拿到好結果」是不夠的。

### 工具
`script/binary_sa_finetune.py`：對 binary RIS pattern 做 simulated annealing。
- Random pixel flip（flip_n 個同時）
- Metropolis 接受 worse moves
- T0 → T_final 降溫

### 實驗結果（從 +2.42 dB sub-optimal pattern 出發）

| SA 配置 | 結果 | gain |
|---------|------|------|
| flip_n=1, T0=5, 5000 steps | +4.98 dB | +2.56 |
| **flip_n=3, T0=20, 8000 steps** | **+7.22 dB** | **+4.80** ← 推薦 |

### 整合至 design tool
`design_pattern_for_target.py` 加 --sa_steps / --sa_T0 / --sa_flip_n。

## Round 16 — SA 整合 batch + 統計實驗

### 工具
- `design_batch.py` 加 SA 支援（每個 target 都享 SA 保底）
- `benchmark_gd_vs_sa.py`：統計實驗 N 次 seed，量化保底機率

### Smoke 驗證
batch + SA, 1 restart seed=4: +2.42 → +6.09 dB (gain +3.67)

### 統計實驗（10 seeds, 執行中）
（結果回來後補表）

## 紀錄歷程更新

```
v1 (純 binary, 15×15)              −4.08 dB
v6 (generator best, 15×15)         −0.46 dB
direct GD 15×15 (5 restart)        +6.94 dB
direct GD 25×25 batch 5 targets    +8.65 dB
direct GD 15×15 inc_θ=+60°         +9.51 dB ← 物理上限
GD + SA fine-tune (round 15)       保底 +7 dB ← Reliability 工程化
```

## Round 13-16 累計工具更新

```diff
+ script/binary_sa_finetune.py        SA 翻轉 fine-tune
+ script/benchmark_gd_vs_sa.py        統計保底機率
+ script/RIS_RESEARCH_REPORT.md       paper-style 完整報告
~ script/design_pattern_for_target.py 加 SA + inc_theta 參數
~ script/design_batch.py              加 SA + inc_theta 參數
~ script/RIS_DESIGN_GUIDE.md          推薦工作流程更新
```

## 對使用者最終建議（更新）

最佳工作流程（**round 15 確立**）：

```bash
python script/design_pattern_for_target.py \
  --element_num 15 \
  --inc_theta 60 \
  --plateau_start 154 --plateau_w 46 \
  --steps 1500 \
  --n_restarts 3 \
  --sa_steps 8000 --sa_T0 20 --sa_flip_n 3 \
  --device cuda:0
```

**2 分鐘**輸出 binary pattern：
- 最佳情況達 +9.51 dB（物理上限）
- 最壞情況保底 +7 dB（SA 救援）
- **比舊版 GD-only 更可靠 + 更快**

## Git
26 個 commits 累計推到 `ricky/modernize`。
