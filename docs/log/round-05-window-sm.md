# Round 5 — 滑動視窗 SM 訓練量（修 R4 的欠訓 + 探測自鎖）

- 狀態: running
- 提出 / 開跑 / 結論: 2026-07-03 / 2026-07-03 / —
- 一句話問題: 把每輪 SM 訓練量從「自鎖在 3–5 epoch」提到「滑動視窗自動找的量級（起點 64、上限 1024）」，能否把 fit_loss 從 ~8–11 壓進 ~1–3 的中間帶、讓 sm_gap/sm_bias 降、trust 進入利用？
- 一句話結論: —（待跑）
- 指向: `configs/README.md`（single_r5_*）· 對照 = R4 同臂（[round-04](round-04-adaptive-sm.md)）· `docs/discuss/decisions.md`「滑動視窗」· memory [[project_sm_training_redesign]]

## 1. 假設 (Propose)
- **問題**: R4 兩個實測發現——① **深度欠訓**：每輪訓完 elite 的訓練 loss 仍停在 7.7–10.6（學長壓到 0.1，差兩個數量級），連訓練集都沒擬合；② **adaptive 探測自鎖**：target 停 3–5、探測曲線 80–100% 平坦（快照全擠低處 → 差異小於雜訊 → 永遠沒有「往上」的證據）。
- **假設**: 訓練量的正確答案在「1–5（欠訓）」與「壓到 0.1（R1 實測過擬合最差）」之間的**從未測過的中間帶**（fit_loss ~1–3，估 16–64+ epoch）。滑動視窗（Ricky 設計）讓訓練量有證據自己爬：每輪訓到視窗頂 → 探測永遠涵蓋上緣 → 無自鎖。
- **機制**（`mode: adaptive_window`，`WindowSMTrainController`）: 每輪 elite 訓滿視窗頂 hi、沿 log2 階梯 [hi/16…hi] 快照 member0；下一輪 held-out 點評快照 → bucket EMA → argmin **區位**決定滑動——連續 3 次落「**上二階**」（hi 或 hi/2，不必貼頂；Ricky：最佳點上方保留至少兩階冗餘）→ hi×2、落「最低一階」→ hi÷2、中段 → 不動。×2 滑動使階梯 key 跨視窗重疊 4/5 → EMA 沿用。已知代價：live SM 過衝最佳點（等衡時 hi≈4–8×argmin，偏多訓、偏成長——與「別回 0.1」之間由視窗上限擋住）。
- **依據**: R4 fit_loss 實錘（2026-07-03，見 `docs/discuss/scratch.md`）；R1（壓到收斂最差 → 別回 0.1）；[[project_sm_training_redesign]]。

## 2. 實驗設計 (Design)
續 R4 factorial，三臂改動：`sm_train.mode: adaptive → adaptive_window` + `ensemble 5→3`（使用者定，省 SM 成本）+ `replay_size 256→512`。
**⚠ 歸因注意**：R5 vs R4 是**兩個實質變更**（視窗訓練量＋ensemble 縮編）——結論要寫整包，不確定性估計（trust/κ 用的成員分歧）在 3 成員下略粗，若 trust 行為異常先想到這個。

| 臂 | config | = R4 同臂改什麼 | 對照 baseline |
| — | — | — | — |
| E 探索 | `single_r5_explore` | `adaptive → adaptive_window` | R4 E（target 自鎖 3-5、gap 反升） |
| D DIP | `single_r5_dip` | 同上 | R4 D（bias ~8、stall 154） |
| E+D | `single_r5_dip_explore` | 同上 | R4 E+D（**破紀錄 -2.89@154**，探索撞到） |

- window 旋鈕（三臂相同）: `snapshots 5 / epoch_min 8 / epoch_max 1024 / hi_init 64 / ema 0.3 / patience 3`（上限 1024＝使用者定，2026-07-03 由 256 調高——爬到頂≈學長「破千」量級）；`ensemble 3`。
- **判準**（分層）: ① `sm_fit_loss` 壓到 ~1–3（機制生效的直接證據）→ ② `sm_gap`/`sm_bias` 降、`trust_t` 升離 0.05（SM 可信）→ ③ worst_margin vs R4 同臂（真目標）。輔看 `sm_train_epochs`（hi 軌跡：爬到哪、有沒有震盪）與 `probe_argmin`（最佳落點）。
- **成本**（單位澄清：1 epoch＝把 elite 整包過一遍＝elite_n 步，batch_size=None 一筆一步）: R4 實測 elite ~90–120 → 起點 hi=64 ≈ 64×115×3 ≈ **2.2 萬步/輪（幾十秒）**沒問題；**爬到頂 hi=1024 ≈ 35 萬步/輪（可能 5–10 分,與 HFSS 同量級）**——**正式機務必盯 `time` 欄**（[[feedback_profile_on_prod_real_hfss]]）；失控就把天花板降 256/512（一行 config）。
- **HFSS 預算**: 各 500 epoch；機器沿用 E@216 / D@37 / E+D@218。

## 3. 執行紀錄 (Run)
| 臂 | 機器 | 狀態 / 進度 | 結果夾 |
| — | — | — | — |
| E | 216 | running（2026-07-03 發） | `[Patch-single-216-43e98b] pixel_single_r5_explore` |
| D | 37 | **提早收**（2026-07-03，@~34ep） | `[Patch-single-37-69f1d3] pixel_single_r5_dip` |
| E+D | 218 | running（2026-07-03 發） | `[Patch-single-218-b817c3] pixel_single_r5_dip_explore` |
- 事件: 2026-07-03 實作 `adaptive_window`（controller + 接線 + 測試，golden 零漂移）→ 同日修正滑動規則為**區位制**（argmin 落上二階即 ×2，不必貼頂——保留冗餘）、上限 256→1024、ensemble 5→3 → **同日發車**（三臂 config 快照已驗證為新版）。
- 事件: 2026-07-03 修 `_resolve_run` 子字串 bug（round 報告兩臂數字相同的元凶；R3 D 數字已更正）。
- 事件: 2026-07-03 晚 **冷啟動超衝止血**——健檢（~22-32ep）發現 sigmoid 兩臂視窗 4 連滑直衝 1024（elite 僅 ~12 筆＝純背誦；fit 0.17-0.91 過擬合、gap 不動、D 30分/ep）→ `epoch_max` 1024→256 重啟三臂（`seed_target` 自動把 hi 夾回、bucket 重學亦順帶解 E 臂 EMA 滯後）。**根因（視窗上限未與 elite 規模掛鉤）留待 R5.5 治本**，詳見 scratch「R5 健檢」交接塊。判讀曲線注意 ep≤重啟點為 1024-cap 段。
- 事件: 2026-07-03 晚 **D 臂提早收**（status 掃描 21:27：D wm最佳 **-7.66**@~34ep 三臂墊底，且 D-only 的隔離問題 R3/R4 已兩度回答＝科學價值最低）→ 釋放 .37 機器給「**除塵驗證**」（harvest 池達標 pattern 拔 1-3px 粉塵後 HFSS 重驗＋順收 rad——戰略討論見 `docs/discuss/scratch.md` 2026-07-03 塊；正式化時開新 round）。R5 續跑 E / E+D 兩臂；round report 收檔時 D 以部分資料入列。

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
