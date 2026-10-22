# Round 94 — v6a +0.15 平台：前段多設計互相移植

- **狀態**: running（mg22 起）
- **提出 / 開跑 / 結論**: 2026-10-22 / 2026-10-22 / —
- **一句話問題**: 前段已有十餘筆 +0.14～+0.15、改動位置各異；以 R89 的區塊移植（底＝前 4、捐贈者＝WM ≥ +0.10 中依底卡住的一緣挑 8）能否拼出 > +0.154？
- **一句話結論 (TL;DR)**: 待分析
- **指向**: [round-93](round-93-v6a-thicken.md) · [round-89](round-89-merge.md) · `script/r89_merge.py`

## 2. 實驗設計 (Design)

- mg22：`--base-mode top --bases 4 --donors 8 --donor-wm 0.10 --max-groups 5 --rank small --per-pair 1`（底 mg18_011、mg19_017、mg20_009、mg20_018；140 候選中 24 組各 1 筆，改 1–3 格）。
- **判準（發車前寫死）**：3 批內出現單次 ≥ +0.18 且換機公證通過 → 成立；最佳 ≤ +0.16 → 不成立，v6a 收尾、整理交付。硬上限 3 批。

## 3. 執行紀錄 (Run)

- `python -m script.r89_merge --n 24 --base-mode top --bases 4 --donors 8 --donor-wm 0.10 --max-groups 5 --rank small --per-pair 1 --name r81v6mg22 --out <dir>`（prio −5）

## 4. 分析 (Analyze)

## 5. 結論 (Conclude)

## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
