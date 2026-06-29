# Round 01 — SM 訓練量 A/B

- **狀態**: archived
- **提出 / 開跑 / 結論**: 2026-06-26 / 2026-06-27 / 2026-06-28
- **一句話問題**: SM 每輪該訓多用力?輕(dlf)/ 訓菁英到飽(dlf_fit)/ 訓全部到飽(refit)?
- **一句話結論 (TL;DR)**: **SM 訓練量不是 bottleneck** — dlf(−4.18)≈refit(−4.21) > dlf_fit(−5.58)、三者皆差 spec ~4dB、搜尋未收斂。
- **指向**: configs/README「SM 訓練量 A/B」三列 · 結果夾 §3 · memory [[project_sm_training_redesign]] [[project_benchmark_vs_random]] · 設計文件 docs/research_landscape.md

## 1. 假設 (Propose)
- **問題 / 假設**: 現行 `dlf` 只訓 elite 1 epoch = under-trained;學長原版訓到 fit。若把 SM 訓到飽,guided 搜尋會不會就贏過 random?
- **為何現在做**: B1 抽驗已確認 `sm_harvest` 對得上現在 HFSS(中位 MSE 1.56)→ SM 來源可信 →「訓練強度」是輸 random 的頭號嫌疑。
- **預期結果與判準**: 若訓練量是病灶 → dlf_fit/refit 的 worst-margin 曲線應顯著高於 dlf。判準看 worst-margin 與 `gap_ema`(對新點準度),**不是 training loss**。
- **依據**: memory [[project_sm_training_redesign]];configs/README「SM 線上更新」段。

## 2. 實驗設計 (Design)
| 臂 | config | 機器 | 唯一變因 | 對照 |
|---|---|---|---|---|
| A `dlf` | `single_guided_harvest` | 216 | `sm_train.mode: dlf`(elite 訓 1 ep + 最新 50 步) | random best-of-N |
| B `dlf_fit` | `single_guided_dlffit_harvest` | 37 | `dlf → dlf_fit`(只訓 elite、訓到收斂) | A |
| C `refit` | `single_guided_refit_harvest` | 218 | `dlf_fit → refit`(訓整個 buffer、含爛 pattern) | B |
- **判準**: worst-margin(dB) vs HFSS-call best-so-far 曲線 + 對比 random best-of-N。
- **HFSS 預算**: 原 ~250 epoch(中途全域改跑到 500;實際各跑到 ~300 才收/crash)。

## 3. 執行紀錄 (Run)
| 臂 | 機器 | 狀態 / 進度 | 結果夾 (NAS) |
|---|---|---|---|
| A `dlf` | 216 | ~304ep,正常跑動 | `T:\…\result\[Patch-single-216-b13433] pixel_single_guided_harvest\` |
| B `dlf_fit` | 37 | ~248ep,**simulator crash** | `…[Patch-single-37-c30f70] pixel_single_guided_dlffit_harvest\` |
| C `refit` | 218 | ~303ep,停/crash | `…[Patch-single-218-c745ee] pixel_single_guided_refit_harvest\` |
- **全域變更 (2026-06-28)**: ① 驗證預算改跑到 500 epoch。② **回滾機制移除**(對 generator-free + K 候選 + 線上 SM 不合身,且原實作有 off-by-one + 覆蓋最佳檔兩個 bug → 實際 ≈ no-op;Round 1 的「不收斂」有它一份)。

## 4. 分析 (Analyze)
`python -m script.round_report --round 01 --runs single_guided_harvest single_guided_dlffit_harvest single_guided_refit_harvest --labels dlf dlf_fit refit --at 250`

| 臂 | 最佳 worst_margin | 達到 epoch |
|---|---|---|
| dlf | −4.18 dB | 297 |
| refit | −4.21 dB | 296 |
| dlf_fit | −5.58 dB | 157 |

- 最佳 pattern + S11/Gain:`assets/round-01/dlf_best.png`、`refit_best.png`、`dlf_fit_best.png`
- worst-margin vs HFSS-call 疊圖:`assets/round-01/benchmark.png`(三臂皆在 spec 線下;無 random 線 — `harvest_single_random` 不存在,待補 random-sim 資料集才能畫對照)
- **觀察**: 三者皆差 spec ~4dB、未達標、搜尋未收斂(最佳是運氣單點;後20均 worst_margin 仍 −7~−10)。dlf_fit **ep157 後 plateau = 最差**(過擬合:訓飽 elite → sm_gap 反升);dlf 與 refit 還在慢慢往上(最佳出現在 ep296-297)。天線有共振(S11 有下凹)但太窄/偏頻,壓不滿 in-band。

## 5. 結論 (Conclude)
- **學到什麼**: **SM 訓練量非 bottleneck**。把 SM 訓更飽不會讓 guided 搜尋贏過 random;訓飽 elite(dlf_fit)反而過擬合最差。
- **決策**: 不再加碼訓練量;轉向文獻治本 = 不確定性 / 信任域門控 + active learning。
- **促成候選**: [[project_litreview_direction]] 升主線 → Round 2(ensemble + trust)。順帶發現:① pattern 太碎(r_feed 0.2 vs sigmoid 0.62)→ 連通先驗候選;② sim_loss 最低 ≠ 天線最好 → loss 對齊 worst_margin 候選。

## 6. 後續決策 (Next)
- **解鎖**: → Round 02(ensemble + trust);baseline 直接用本 round 的 A(dlf)、C(refit),不重跑控制組。
- **新待辦**(已進 ONGOING 🔜): DIP/連通 sc↑ / loss 對齊 worst_margin / val-早停 / 可解釋性 / mirror loss / harvest 重錨。

## 7. 歸檔指向 (Archive)
- configs/README 列: `single_guided_harvest` / `single_guided_dlffit_harvest` / `single_guided_refit_harvest`
- 結果夾: 見 §3
- memory: [[project_sm_training_redesign]] [[project_benchmark_vs_random]] [[project_litreview_direction]] [[project_generator_hyperfeature_pivot]]
- 設計文件: docs/research_landscape.md、docs/guided_search_design.md
- ONGOING 動作: 從 🔵 移除,✅ 區指向本檔。
