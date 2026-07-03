# Round 4 — 自適應 SM 訓練量（修 R3 的 SM 欠訓瓶頸）

- 狀態: archived  <!-- 2026-07-03 提早停(E@208/D@222/E+D@201,未到 500)釋放機器給 Round 5 -->
- 提出 / 開跑 / 結論: 2026-07-02 / 2026-07-02 / 2026-07-03
- 一句話問題: 用「held-out 的 fresh HFSS 點」自調每輪 SM 重訓 epoch 數，能否修掉 R3 的「SM 欠訓 → trust 鎖死 → plateau」，讓 E/D/E+D factorial 終於乾淨？
- 一句話結論: **E+D 破專案紀錄 -2.89@154**（探索躍遷、非 trust 利用）；但主假設未驗證——adaptive v1 探測自鎖在 3-5 epoch、SM 仍深度欠訓（fit_loss 8-11）、trust 全程鎖 0.05；E/D 皆輸 R3 同臂 ~0.9dB → **Round 5 滑動視窗**修探測結構
- 指向: `configs/README.md`（single_r4_*）· 結果夾（待）· memory [[project_sm_training_redesign]] · `docs/discuss/decisions.md`「自適應 SM 訓練量」

## 1. 假設 (Propose)
- **問題 / 假設**: R3 健檢發現三臂（E/D/E+D）被同一個上游瓶頸汙染——`mode:dlf` 每輪 elite 只訓 **1 epoch** → SM 欠訓、系統性低估 HFSS ~3–4.5 → `trust_t` 因 gap 高被鎖在 0.05、永不進入利用 → 早期好解(best@ep8/89)被純探索沖走、plateau。假設：**自調 SM 訓練量到「泛化最好」**（非固定 1 epoch、也非逼 min_loss=0.1 過擬合）→ gap 降 → trust 升 → 進入利用 → 破 plateau。
- **為何現在做**: R3 已定位瓶頸在 SM 訓練強度＋利用不足（非探索量、非連通）；與使用者討論定案「自適應訓練量」機制（見 `docs/discuss/decisions.md`）並實作完成（opt-in `mode:adaptive`、golden 零漂移）。這也實現了 ONGOING 舊候選「val-早停」。
- **預期結果與判準**: 相對 R3 同臂，看 **worst_margin** 有沒有變好（往 0/正）、**trust_t** 有沒有升離 0.05、**sm_bias**（新增訊號）有沒有降；輔以 `sm_train_epochs`（自適應訓練量收斂到多少）、`probe_argmin`（探測曲線最佳點）。
- **依據**: R3 健檢（sm_fit_epochs 全程=1、sm_bias +3~4.5、trust_t 鎖 0.05）；文獻方向 [[project_litreview_direction]]；[[project_sm_training_redesign]]（1 epoch under-trained、逼收斂過擬合）。

## 2. 實驗設計 (Design)
factorial 續 R3（探索×DIP），唯一新變因＝三臂全把 `sm_train.mode: dlf → adaptive`（移除 R3 的 SM 欠訓 confound）。

| 臂 | config | = R3 同臂改什麼 | 對照 baseline |
| — | — | — | — |
| E 探索 | `single_r4_explore` | `mode: dlf→adaptive`（direct 8 候選、lr 0.015 不變） | R3 E（固定 dlf） |
| D DIP | `single_r4_dip` | `mode: dlf→adaptive`（sigmoid、lr 0.005 不變） | R3 D（固定 dlf） |
| E+D | `single_r4_dip_explore` | `mode: dlf→adaptive`（sigmoid、lr 0.015 不變） | R3 E+D（固定 dlf） |

- adaptive 旋鈕（三臂相同）：`snapshots 5 / epoch_min 1 / epoch_max 32 / ema 0.3`；ensemble 保持 5、探測只用 member0。
- **⚠ 歸因注意（兩個機制）**：`mode: dlf→adaptive` 實際換了兩件事——elite 訓練量 1→自適應、**且拿掉對最新點的
  `train_one_data` 單筆擬合**（反模式；最新點在 elite 裡照訓）。R4 vs R3 的結論要寫「SM 更新規則整包換」，
  不能歸因到單一機制（見 `docs/discuss/decisions.md` 補記）。
- **判準**: worst_margin（真目標）+ trust_t 升 + sm_bias 降；cross-round 比 R3 同臂看自適應有沒有幫上。
- **HFSS 預算**: 各 500 epoch（沿 R3；⚠ 每輪 SM 重訓量會比 dlf 大，正式機量 SM 佔 HFSS 比例）。

## 3. 執行紀錄 (Run)
| 臂 | 機器（沿 R3 配置排除機器差異） | 狀態 / 進度 | 結果夾 |
| — | — | — | — |
| E | 216 | 2026-07-03 停 @208 ep | `[Patch-single-216-b159bb] pixel_single_r4_explore` |
| D | 37（快） | 2026-07-03 停 @222 ep | `[Patch-single-37-0b9ad8] pixel_single_r4_dip` |
| E+D | 218 | 2026-07-03 停 @201 ep（**-2.89@154 破紀錄**） | `[Patch-single-218-…] pixel_single_r4_dip_explore` |
- 事件 / 全域變更: 進 R4 前一批工程（Round4 準備）：新增追蹤訊號（flips/stall/sm_bias/wm_per-label/sm_train_epochs/probe_*/elite_n）；實作 opt-in 自適應-SM（golden 零漂移）。發前先跑 `python -m script.status` 掃機器真相。
- 2026-07-02 發車前健檢：修控制器「低訓練量死鎖」（float target + 本輪觀測投票，模擬實證；沒修的話 R4 很可能默默退化回 dlf、整輪白跑）＋ `seed_target()` 斷點續跑續 target ＋ `elite_n` 成本訊號。詳見 `docs/discuss/decisions.md` 補記。
- 2026-07-02 **更正**：三臂**從起跑即為修復版**（使用者發車時注意到修復尚未 push、手動把改動帶上正式機；證據＝csv 自首個 epoch 就有 `elite_n`，該欄與死鎖修復同 commit）。先前「舊碼起跑、E 臂死鎖前兆」為誤判——E 的 `sm_train_epochs` 8→3 下滑是修復版在「探測沒資訊」（`probe_min≈probe_max`、雜訊主導）時的**預期保守行為**（退向低訓練量≈dlf、可爬回），非死鎖。**觀察點**：若某臂長期釘在 epoch_min 且探測曲線持續平 → K=1 探測被雜訊淹沒＝真訊號，屆時再議加探測點/調 ema。
- 2026-07-03 中檢（~200ep）：**E+D 破專案紀錄 -2.89@154**（勝 R3 全程 best +2.80；長停滯後單步躍遷＝探索撞到，非 trust 利用）；sigmoid 兩臂 sm_gap 穩降（線上累積生效）、E 臂 gap 反升（ping-pong 傷 SM）；三臂 trust 仍全鎖 0.05。**實錘深度欠訓**：每輪訓完 elite fit_loss 仍 7.7-10.6（學長壓 0.1）＋探測自鎖 target 3-5 → 催生 **Round 5 滑動視窗**（[round-05](round-05-window-sm.md)，待本輪收檔發）。

## 4. 分析 (Analyze)
2026-07-03 產（`python -m script.round_report --round 04 --runs single_r4_explore single_r4_dip single_r4_dip_explore --labels E D E+D --at 201`；提早停 E@208/D@222/E+D@201，未到 500、釋放機器給 R5）：

| 臂 | 最佳 worst_margin | 達到 epoch |
|---|---|---|
| E+D | **-2.89 dB**（專案史上最佳） | 154 |
| E | -4.58 dB | 111 |
| D | -6.75 dB | 61 |

- **cross-round vs R3 同臂（同 epoch 預算）**：E+D **+2.80**（-2.89 vs -5.69）／E **-0.94**（-4.58 vs -3.63@89）／D **-0.92**（-6.75 vs -5.83）。
- E+D 的躍遷形狀：-7.60@76 → -7.29@135 → -7.13@153 → **-2.89@154**（長停滯後單步 +4.24dB＝探索撞進質變區）；發生時 trust_t 仍鎖 0.05 → **是探索的功勞，不是 trust 利用**。
- **訊號總結**（詳見 §3 各中檢）：三臂 trust 全程鎖 0.05（gap 6.5–9 遠高於門檻）；每輪訓完 elite fit_loss 仍 7.7–10.6（**深度欠訓**）；adaptive 探測自鎖 target 3–5、曲線 80–100% 平坦；E 臂 sm_gap 反升 3→9（兩群 ping-pong、SM 對搜尋區越來越盲）；D 臂 SM 指標三臂最佳（gap 穩降至 6.5、bias 4.6）卻 stall 154——**SM 變好 ≠ 找到更好解**。
- 圖：`assets/round-04/E_best.png` · `D_best.png` · `E+D_best.png` · `benchmark.png`（無 random 線，`harvest_single_random` 仍 0 筆）。

## 5. 結論 (Conclude)
- **主假設未驗證**：「自適應訓練量 → SM 可信 → trust 升 → 利用」這條鏈沒走通——根因是 adaptive v1 只探測「目前訓練量以下」的範圍（結構性自鎖），訓練量從沒真的提上去，SM 始終欠訓。
- **但 R4 不是白跑**：① E+D 破專案紀錄 -2.89（勝 R3 同臂 +2.80）——adaptive 臂的探索比 R3 dlf 更有生產力；② 拿到「深度欠訓」與「探測自鎖」兩個實錘診斷（fit_loss 8-11、target 3-5、曲線平），直接催生 R5 設計；③ 兩機制包的副作用現形——拿掉 train_one_data 後 direct 臂（E）出現兩群 ping-pong、sm_gap 反升（見 §2 歸因注意＋scratch「ping-pong」塊）。
- **sigmoid 臂的線上累積有效**（gap 穩降）＝學長「線上學習要等」成立；direct 臂相反（ping-pong 傷資料品質）。
- **學到什麼**：訓練量問題要「探測永遠看得到上方」才解得掉 → Round 5 滑動視窗（訓到視窗頂、argmin 區位滑動）。

## 6. 後續決策 (Next)
- → **Round 5 滑動視窗**（[round-05](round-05-window-sm.md)）：訓到視窗頂（探測永遠涵蓋上緣、無自鎖）、argmin 落上二階×2／最低階÷2、上限 1024、ensemble 5→3。2026-07-03 已發。
- E 臂 ping-pong → 「選擇端 known-bad 鄰域懲罰」候選續掛 ONGOING（觸發條件不變）。
- E+D 的 -2.89 紀錄 pattern 已安全存 `patterns/`；R5 E+D 看能否把「撞到」變「開採」。

## 7. 歸檔指向 (Archive)
- configs/README 列: `single_r4_explore` / `single_r4_dip` / `single_r4_dip_explore`
- 結果夾: 見 §3 表；圖: `assets/round-04/`；紀錄解 pattern 在 E+D 結果夾 `patterns/`
- memory: [[project_sm_training_redesign]] · [[project_discussion_memory]]
- 設計文件: `docs/discuss/decisions.md`「自適應 SM 訓練量」＋「滑動視窗」
- 接棒: [round-05](round-05-window-sm.md)（2026-07-03 已發）
