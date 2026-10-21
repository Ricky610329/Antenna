# Round 91 — 切回 v6（−20）：實測導向小步工具上 v6

- **狀態**: running（`r81v6w01` 起）
- **提出 / 開跑 / 結論**: 2026-10-22 / 2026-10-22 / —
- **一句話問題**: v6a 以實測導向小步（移植、同底疊加、擾動後修復、`--predict`）從 −0.20 推到 +0.147；同一套工具改用 v6 尺（阻帶 −20），能否推動停在 −3.11（b92_016，b92 起無進展）的 v6？
- **一句話結論 (TL;DR)**: 待分析
- **指向**: [round-90](round-90-reach-plus01.md) · [round-81](round-81-wide-filter.md)（v6 規格、通帶天花板分析）· `script/r89_merge.py --config configs/dual_r81_v6.yaml`

## 1. 假設 (Propose)

- v6 前段主要卡低阻 14.5–15 GHz（需 ≤ −20，現約 −16.9），其次通帶與 S11。v6a 的經驗顯示實測導向的小步比模型選樣有效；用 v6 尺的單格翻轉實測統計（親代放寬到 WM > −6）預估五項 margin，可找到推動低阻的步。

## 2. 實驗設計 (Design)

- w01：`--predict` 底 v6 前 3（b92_016 −3.11、b94_000 −3.33、b94_015 −3.39），24 筆（兩兩 18、單格 6），預估 −2.6～−3.2。
- 後續（依結果）：v6 前段的區塊移植、同底已驗證單步疊加；R85 步驟 B（交接檔低阻單格槓桿 (2,7)(3,9)(6,9)(4,15)(17,10)(22,16)）。
- **判準（發車前寫死）**：3 批內 v6 最佳 ≥ −2.8（較 −3.11 進 0.3 dB 以上）→ 成立；≥ −3.0 → 回報；< −3.0 → 不成立、評估非 a0 探索輪（ONGOING 候選）或與 Ricky 討論規格。紀錄級以 216／37 驗收。
- 硬上限 3 批。

## 3. 執行紀錄 (Run)

- w01：`python -m script.r89_merge --config configs/dual_r81_v6.yaml --n 24 --base-mode top --stats-min-parent -6 --predict r81v2_b92_016_1921041c,r81v2_b94_000_3dc4d025,r81v2_b94_015_8aaaa10e --name r81v6w01 --out <dir>`
- 入佇列：`python -m script.r81_curve_sm enqueue --input <dir> --name r81v6w01 --prio -3 --shards 3 --config configs/dual_r81_v6.yaml`
- w02：`python -m script.r89_merge --config configs/dual_r81_v6.yaml --n 24 --base-mode top --levers "2,7;3,9;6,9;4,15;17,10;22,16" --lever-base r81v2_b92_016_1921041c,r81v2_b94_000_3dc4d025 --name r81v6w02 --out <dir>`

## 4. 分析 (Analyze)

### 2026-10-22 05:16：w01 前 13 筆——v6 低阻不動；w02 低阻槓桿（R85 步驟 B）發車

- w01（`--predict`，v6 尺）13／24：最佳 −3.047（b92_016＋(23,20) 加銅＋(8,19) 挖空；底 −3.107），**低阻全停在 −2.85～−3.43**；預估偏樂觀 0.2～3 dB。單格翻轉統計主要來自 v6a 前段（低阻非瓶頸），幾乎沒有推動 14.5 GHz 的樣本。
- w02（prio −3）：`r89_merge --levers "2,7;3,9;6,9;4,15;17,10;22,16" --lever-base b92_016,b94_000`——交接檔的低阻單格槓桿，單步 10＋兩兩 14。看點：低阻能否推過 −2.5，代價落在哪一項（交接檔觀察：高阻 −4 dB 級）。新增 `lever_candidates`／`lever_main`，附測試。

## 5. 結論 (Conclude)

## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
