# Round 95 — v6a 續爬：+0.20 之後（移植／預估交替）

- **狀態**: running（mg25 起）
- **提出 / 開跑 / 結論**: 2026-10-22 / 2026-10-22 / —
- **一句話問題**: 公證最佳 +0.202 之後，移植與 `--predict` 交替能否再往上推？
- **一句話結論 (TL;DR)**: 待分析
- **指向**: [round-94](round-94-plateau-merge.md) · `script/r89_merge.py`

## 2. 實驗設計 (Design)

- mg25：移植（底 mg23_018、mg23_013、mg20_011；捐贈者 WM ≥ +0.14，15 筆）。mg26：`--predict`（底 mg23_018、mg23_013、mg25_007，24 筆）。mg27：依結果。
- **判準（發車前寫死）**：3 批內公證 ≥ +0.25 → 成立；+0.20～+0.25 → 回報；未破 +0.202 → 不成立（v6a 收尾整理交付）。硬上限 3 批。
- 公證：新高換機（避開與原量同機；218 對部分幾何離群，優先 216／37）。

## 3. 執行紀錄 (Run)

- mg25：`r89_merge --n 24 --base-mode top --bases 3 --donors 8 --donor-wm 0.14 --max-groups 5 --rank small --per-pair 1 --name r81v6mg25`
- mg26：`r89_merge --n 24 --base-mode top --predict r81v6mg23_018_d4347f75,r81v6mg23_013_70f5551f,r81v6mg25_007_7e234ed1 --name r81v6mg26`

## 4. 分析 (Analyze)

## 5. 結論 (Conclude)

## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
