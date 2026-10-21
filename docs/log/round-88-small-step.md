# Round 88 — 利用／探索臂只准小步（離親代 ≤ 4 格）

- **狀態**: running（掛在 R81 v6a 批次內，b112 起）
- **提出 / 開跑 / 結論**: 2026-10-21 / 2026-10-21 / —
- **一句話問題**: 近頂端時翻 ≥ 5 格的子代從未改善（R87 §4：29 筆 0 筆），利用／探索臂若只從離親代 ≤ 4 格的候選挑，能否少浪費名額、提高前段命中？
- **一句話結論 (TL;DR)**: 待分析
- **指向**: [round-87](round-87-ctx-climb.md) §4 · `script/r81_curve_sm.py --small-exploit/--small-explore/--small-d`

## 1. 假設 (Propose)

- v6a 前段（WM −0.25～−0.6）處於精修階段，單步最大增益約 +0.14 dB；大步變異（含推移交界）幾乎必然破壞已達標的四項。限制步長的臂應有更高中位與更少大失手。

## 2. 實驗設計 (Design)

| 臂 | 名額 | 候選池 | 打分 |
|---|---|---|---|
| tclimb | 10 | 爬山鄰居（1–3 格） | 上下文 − 0.5·CNN std |
| **eexploit_s** | 4 | 離親代 ≤ 4 格 | 平均分 LCB |
| eexploit（對照） | 3 | 全部 | 平均分 LCB |
| **dexplore_s** | 2 | 離親代 ≤ 4 格 | CNN UCB |
| dexplore（對照） | 2 | 全部 | CNN UCB |
| random | 3 | 全部 | 均勻 |

- 小步臂先挑、再由剩餘候選挑對照臂（同批）。
- **判準（發車前寫死）**：b112–b114 三批合計，小步臂（*_s 合併，約 18 筆）實測中位 ≥ 對照臂（exploit＋explore 合併，約 15 筆）中位 + 0.2 dB，且最佳 ≥ 對照最佳 − 0.1 → 成立（exploit／explore 全改小步）；小步中位低於對照中位 → 撤回；其間 → 回報。以 v6a 尺計（中途若公證達標切回 v6，改以 v6 尺並註記）。
- 硬上限 3 批。
- **2026-10-21 08:22 注記（發生在任何結果回來之前）**：b112 建批後量步長，對照臂 5 筆中只有 1 筆 > 4 格（2、3、3、4、8 格）——模型本來就偏好小步，兩組重疊大、主判準鑑別力偏低，預期多半落在「回報」區。加一項次要分析：對照臂中 > 4 格的子代單獨與小步臂比（中位、改善率）。主判準不改。

## 3. 執行紀錄 (Run)

- `python -m script.r81_curve_sm build --batch N --n 24 --climb-n 10 --delta-climb 10 --climb-main ctx --model-arms ens --explore-model cnn --small-exploit 4 --small-explore 2 --small-d 4 --delta-exploit --delta-explore --ctx-shadow --climb-top 10 --max-flips 12 --exploit-lcb 0.5 --parent-score violation --p-cross 0.3 --disconnect --push 0.5 --reseam 0 --config configs/dual_r81_v6a.yaml --out <dir>`
- 入佇列：`python -m script.r81_curve_sm enqueue --input <dir> --name r81v6bN --prio -2 --shards 3 --config configs/dual_r81_v6a.yaml`

## 4. 分析 (Analyze)

## 5. 結論 (Conclude)

## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
