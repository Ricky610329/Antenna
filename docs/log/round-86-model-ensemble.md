# Round 86 — CNN 變化量＋上下文接龍模型平均打分

- **狀態**: running（掛在 R81 v6a 批次內，b101 起）
- **提出 / 開跑 / 結論**: 2026-10-20 / 2026-10-20 / —
- **一句話問題**: 影子模式六批中，上下文接龍（Transformer，可變 K）與 CNN 變化量模型準度互有勝負、誤差相當；兩者平均打分能否比單用 CNN 挑得更好？
- **一句話結論 (TL;DR)**: 待分析
- **指向**: [round-83](round-83-delta-exploit.md)（影子三批）· [round-84](round-84-delta-explore.md) §4 · `script/r81_curve_sm.py --ens-climb`

## 1. 假設 (Propose)

- 兩模型結構不同（K=1 CNN 與 K=4 注意力合成），錯誤不完全相關；平均可降低單一模型的偏差，提升前段排序。

## 2. 實驗設計 (Design)

| 臂 | 名額 | 打分 |
|---|---|---|
| eclimb | 爬山 5 | (CNN 均值 + 上下文模型) ÷ 2 − 0.5·CNN std |
| dclimb（同批對照） | 爬山 5 | CNN 均值 − 0.5·CNN std |

- 兩份排序交替取（重複者歸先取到的一方），避免一方先挑走另一方的前段。exploit／explore／random 不變。
- **判準（發車前寫死）**：b101–b103 三批合計，eclimb 的實測中位 ≥ dclimb 中位，且最佳 ≥ dclimb 最佳 −0.1 dB → 成立（其他臂也改用平均）；中位比 dclimb 差 0.3 dB 以上 → 撤回；其間 → 回報。以 v6a 尺計（若中途切回 v6，改以 v6 尺並註記）。
- 硬上限 3 批。

## 3. 執行紀錄 (Run)

- `python -m script.r81_curve_sm build --batch N --n 24 --climb-n 10 --delta-climb 10 --ens-climb 5 --delta-exploit --delta-explore --ctx-shadow --climb-top 10 --max-flips 12 --exploit-lcb 0.5 --parent-score violation --p-cross 0.3 --disconnect --push 0.5 --reseam 0 --config configs/dual_r81_v6a.yaml --out <dir>`

- 2026-10-20 21:02：首建 b101 時兩臂各 5 筆**全落在同一個親代**（整份排序交替取，而 pick_climb 把最好親代的一半名額排在最前）。發車前撤下（未被認領，NAS input 夾與 jobs 條目已刪），改 `pick_climb_pair`：兩臂各自照 climb_shares 分親代名額、同親代內交替取；補回歸測試後重建 b101。

## 4. 分析 (Analyze)

## 5. 結論 (Conclude)

## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
