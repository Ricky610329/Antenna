# Round 5 — 滑動視窗 SM 訓練量（修 R4 的欠訓 + 探測自鎖）

- 狀態: proposed
- 提出 / 開跑 / 結論: 2026-07-03 / — / —
- 一句話問題: 把每輪 SM 訓練量從「自鎖在 3–5 epoch」提到「滑動視窗自動找的量級（起點 64、上限 1024）」，能否把 fit_loss 從 ~8–11 壓進 ~1–3 的中間帶、讓 sm_gap/sm_bias 降、trust 進入利用？
- 一句話結論: —（待跑）
- 指向: `configs/README.md`（single_r5_*）· 對照 = R4 同臂（[round-04](round-04-adaptive-sm.md)）· `docs/discuss/decisions.md`「滑動視窗」· memory [[project_sm_training_redesign]]

## 1. 假設 (Propose)
- **問題**: R4 兩個實測發現——① **深度欠訓**：每輪訓完 elite 的訓練 loss 仍停在 7.7–10.6（學長壓到 0.1，差兩個數量級），連訓練集都沒擬合；② **adaptive 探測自鎖**：target 停 3–5、探測曲線 80–100% 平坦（快照全擠低處 → 差異小於雜訊 → 永遠沒有「往上」的證據）。
- **假設**: 訓練量的正確答案在「1–5（欠訓）」與「壓到 0.1（R1 實測過擬合最差）」之間的**從未測過的中間帶**（fit_loss ~1–3，估 16–64+ epoch）。滑動視窗（Ricky 設計）讓訓練量有證據自己爬：每輪訓到視窗頂 → 探測永遠涵蓋上緣 → 無自鎖。
- **機制**（`mode: adaptive_window`，`WindowSMTrainController`）: 每輪 elite 訓滿視窗頂 hi、沿 log2 階梯 [hi/16…hi] 快照 member0；下一輪 held-out 點評快照 → bucket EMA → argmin 連續 3 次貼頂 → hi×2、貼底 → hi÷2、中間 → 不動。×2 滑動使階梯 key 跨視窗重疊 4/5 → EMA 沿用。已知代價：live SM 最多過衝一個 octave。
- **依據**: R4 fit_loss 實錘（2026-07-03，見 `docs/discuss/scratch.md`）；R1（壓到收斂最差 → 別回 0.1）；[[project_sm_training_redesign]]。

## 2. 實驗設計 (Design)
續 R4 factorial，三臂唯一變因 `sm_train.mode: adaptive → adaptive_window`（+ `replay_size 256→512`）。

| 臂 | config | = R4 同臂改什麼 | 對照 baseline |
| — | — | — | — |
| E 探索 | `single_r5_explore` | `adaptive → adaptive_window` | R4 E（target 自鎖 3-5、gap 反升） |
| D DIP | `single_r5_dip` | 同上 | R4 D（bias ~8、stall 154） |
| E+D | `single_r5_dip_explore` | 同上 | R4 E+D（**破紀錄 -2.89@154**，探索撞到） |

- window 旋鈕（三臂相同）: `snapshots 5 / epoch_min 8 / epoch_max 1024 / hi_init 64 / ema 0.3 / patience 3`（上限 1024＝使用者定，2026-07-03 由 256 調高——爬到頂≈學長「破千」量級）。
- **判準**（分層）: ① `sm_fit_loss` 壓到 ~1–3（機制生效的直接證據）→ ② `sm_gap`/`sm_bias` 降、`trust_t` 升離 0.05（SM 可信）→ ③ worst_margin vs R4 同臂（真目標）。輔看 `sm_train_epochs`（hi 軌跡：爬到哪、有沒有震盪）與 `probe_argmin`（最佳落點）。
- **成本**: 視窗爬到頂時 hi=1024 × elite 數百點 ≈ **數十萬步/輪**——這已不再必然可忽略，**正式機務必盯 `time` 欄**看 SM 佔 wall-clock 比例（[[feedback_profile_on_prod_real_hfss]]）；若佔比失控，天花板降回 256/512 是一行 config 的事。
- **HFSS 預算**: 各 500 epoch；機器沿用 E@216 / D@37 / E+D@218。

## 3. 執行紀錄 (Run)
| 臂 | 機器（計畫） | 狀態 / 進度 | 結果夾 |
| — | — | — | — |
| E | 216 | proposed（待 R4 收檔） | — |
| D | 37 | proposed（待 R4 收檔） | — |
| E+D | 218 | proposed（待 R4 收檔） | — |
- 事件: 2026-07-03 實作 `adaptive_window`（controller + 接線 + 測試，golden 零漂移）；config ready。**發車前**: `python -m script.status` 確認 R4 已停、正式機 `git pull`。

## 4. 分析 (Analyze)
（待跑完 `python -m script.round_report --round 05 --runs single_r5_explore single_r5_dip single_r5_dip_explore --labels E D E+D`）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 若 fit_loss 壓下去但 wm 沒變好 → 訓練量非殘餘瓶頸 → 回頭看選擇端（known-bad 懲罰候選）/ loss 對齊。
- 若 hi 震盪不收斂 → probe 雜訊仍主導 → 上 M 點平均（probe round 設計，見 scratch）。
- 若 R4 E+D 的紀錄區在 R5 被穩定開採 → 「撞到→開採」機制成立，值得寫進論文敘事。

## 7. 歸檔指向 (Archive)
- configs/README 列: `single_r5_explore` / `single_r5_dip` / `single_r5_dip_explore`
- 結果夾: （待）
- memory: [[project_sm_training_redesign]] · [[project_discussion_memory]]
- 設計文件: `docs/discuss/decisions.md`「滑動視窗 SM 訓練量」
- ONGOING 動作: 發車時移 🔵、跑完移 ✅
