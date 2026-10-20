# Round 87 — 爬山打分：平均 vs 上下文接龍模型單獨

- **狀態**: running（掛在 R81 v6a 批次內，b107 起）
- **提出 / 開跑 / 結論**: 2026-10-21 / 2026-10-21 / —
- **一句話問題**: R86 前段準度顯示 CNN 變化量模型在好設計之間已無排序力（ρ −0.13），上下文接龍模型 +0.55、平均 +0.45；爬山若只用上下文模型打分，能否比平均挑得更好？
- **一句話結論 (TL;DR)**: 待分析
- **指向**: [round-86](round-86-model-ensemble.md) §4 · `script/r81_curve_sm.py --climb-main/--climb-alt/--model-arms`

## 1. 假設 (Propose)

- 平均打分的增益幾乎全來自上下文模型；在 v6a 前段（WM −0.3～−1），CNN 的排序反而稀釋訊號。上下文模型單獨打分應在爬山臂有更高中位。

## 2. 實驗設計 (Design)

| 臂 | 名額 | 打分 |
|---|---|---|
| eclimb（主，基準） | 爬山 5 | (CNN + 上下文) ÷ 2 − 0.5·CNN std |
| **tclimb（測試）** | 爬山 5 | 上下文 − 0.5·CNN std |
| eexploit | 7 | 平均分 LCB（R86 預登記的後續，非本輪測試對象） |
| eexplore | 4 | 平均分 UCB（同上） |
| random | 3 | 均勻 |

- 兩爬山臂各自照親代分名額、同親代內交替取（`pick_climb_pair`）。
- **判準（發車前寫死）**：b107–b109 三批合計，tclimb 實測中位 ≥ eclimb 中位，且最佳 ≥ eclimb 最佳 − 0.1 dB → 成立（爬山改上下文單獨）；中位比 eclimb 差 0.3 dB 以上 → 撤回；其間 → 回報。以 v6a 尺計（中途若公證達標切回 v6，改以 v6 尺並註記）。
- eexploit／eexplore 只做描述（與 R86 的 dexploit／dexplore 比，非同批，不下因果結論）。
- 硬上限 3 批。

## 3. 執行紀錄 (Run)

- `python -m script.r81_curve_sm build --batch N --n 24 --climb-n 10 --delta-climb 10 --climb-main ens --climb-alt ctx --climb-alt-n 5 --model-arms ens --delta-exploit --delta-explore --ctx-shadow --climb-top 10 --max-flips 12 --exploit-lcb 0.5 --parent-score violation --p-cross 0.3 --disconnect --push 0.5 --reseam 0 --config configs/dual_r81_v6a.yaml --out <dir>`
- 入佇列：`python -m script.r81_curve_sm enqueue --input <dir> --name r81v6bN --prio -2 --shards 3 --config configs/dual_r81_v6a.yaml`

## 4. 分析 (Analyze)

## 5. 結論 (Conclude)

## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
