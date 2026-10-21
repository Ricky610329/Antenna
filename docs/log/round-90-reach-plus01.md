# Round 90 — v6a 推到 +0.1（同底驗證疊加＋上緣專攻）

- **狀態**: running（mg10 起）
- **提出 / 開跑 / 結論**: 2026-10-21 / 2026-10-21 / —
- **一句話問題**: Ricky 要 v6a ≥ +0.1（公證）才切回 v6；目前單次最佳 `mg08_018` +0.080、公證最佳 `mg07_003` +0.063。以實測導向小步（同底已驗證單步組合、30 GHz 上緣翻轉、兩條路線合併）能否在 3 批內越過 +0.1？
- **一句話結論 (TL;DR)**: 待分析
- **指向**: [round-89](round-89-merge.md)（方法來源）· `script/r89_merge.py --stack-same-parent／--edge 30`

## 1. 假設 (Propose)

- 前段兩條路：`mg08_018`（mg05_009＋(0,9)＋(14,12)）與 `mg09_004`（mg05_009＋009＋015＋001），互不衝突的單步可近似相加；排除已知互斥步（mg07_005、b112_007）後組合可越過 +0.1。

## 2. 實驗設計 (Design)

- 批：mg10（mg08_018 上套實測有效單步／兩兩，24）、mg11（mg09_004 上單步，8，prio −5）、mg12（前 2 名上緣專攻，12）——三批即上限。
- **判準（發車前寫死）**：三批內出現單次 WM ≥ +0.1 → 換機公證；公證通過 → 依 Ricky 指示切回 v6、本輪成立。三批後最佳仍 < +0.1 → 回報並評估開非 a0 探索輪或與 Ricky 討論門檻。
- 公證：`notarize_v6a.py`（暫存區）釘 216／prio −5，id `~rep1`、kind repeat。

## 3. 執行紀錄 (Run)

- mg10：`python -m script.r89_merge --n 24 --base-mode top --rank small --stack 14 --stack-k 7 --stack-sizes 1,2 --stack-min-parent -0.05 --stack-base r81v6mg08_018_41808495 --name r81v6mg10`（prio −4）
- mg11：`--n 8 --stack 8 --stack-k 10 --stack-sizes 1 --stack-min-parent -0.05 --stack-base r81v6mg09_004_85f33612 --name r81v6mg11`（prio −5、1 片）
- mg12：`--n 24 --base-mode top --edge 30 --edge-bases 2 --edge-pair 3 --name r81v6mg12`（prio −4，過濾後 12 筆）

## 4. 分析 (Analyze)

## 5. 結論 (Conclude)

## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
