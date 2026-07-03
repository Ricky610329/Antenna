# Round 03 — 探索 × DIP（factorial：E / D / E+D）

- **狀態**: archived  <!-- 2026-07-01 發;2026-07-02 提早停(E@189/D@101/E+D@132,未到 500)釋放機器給 Round 4 -->
- **提出 / 開跑 / 結論**: 2026-07-01 / 2026-07-01 / 2026-07-02
- **一句話問題**: (E) 搜尋凍住(每 epoch 才翻 ~6 像素)→ 加大步長探索能解凍+變好嗎？(D) 把 generator 帶回來(sigmoid 連通先驗)能救 S11 不共振嗎？(E+D) 兩者加乘？
- **一句話結論 (TL;DR)**: **E(lr↑)最佳** -3.63dB@89（約 ¼ epoch 追平 reference ②）；DIP 連通成功（r_feed~0.95）但結果停滯（best@8 後不動）；**三臂被共同瓶頸「SM 欠訓→trust 鎖死→不利用」汙染、factorial 讀不乾淨 → Round 4 修瓶頸重跑**
- **指向**: configs/README「Round 3」三列 · reference = Round-2 ②（[round-02](round-02-ensemble-trust.md)）· memory [[project_generator_hyperfeature_pivot]] [[project_litreview_direction]] · 設計文件 docs/guided_search_design.md

## 1. 假設 (Propose)
- **問題 / 假設**:
  - Round 2 實測 dlf+ensemble 每 epoch 只翻 ~6 像素 = 搜尋**凍住**（ensemble 平均使 SM 地形超穩、步太小；trust 無關、① =② 都 6）。→ **(E)** 加大步長(lr)能否解凍且變好？
  - generator-free 丟掉架構連通先驗（實測 r_feed 0.2 vs sigmoid 0.62）→ 可能是 S11 不共振主因。→ **(D)** 帶回 sigmoid(DIP)+ 保留治本能否救連通/結果？
- **為何現在做**: 承接 Round 2（治本微幅未決定性）+ 使用者的「探索 vs 保守」假設 + 原訂 DIP round。
- **預期結果與判準**: **同時看**「像素翻轉數（探索量）」與 worst_margin（結果）。E vs ② = 探索效果；D vs ② = DIP 效果；E+D vs E/D = 加乘。
- **依據**: 本輪 pattern-volatility 實測（① 6 / ② 6 / ③ 277 翻轉）；[[project_generator_hyperfeature_pivot]]（r_feed 0.2 vs 0.62）。

## 2. 實驗設計 (Design)
**factorial：兩個正交旋鈕 = lr(探索) × generator(DIP)。lr 是唯一對 direct & sigmoid 都適用的探索旋鈕 → 保 factorial 乾淨（UCB/diversity 候選式、sigmoid 用不了 → 之後另做 direct-only 子臂）。**

| 臂 | config | = ② 改什麼 | 隔離變因 |
|---|---|---|---|
| **reference** | `single_r2_enstrust_harvest`（②，已有數據） | — | dlf+ensemble+trust、direct、lr 0.005 |
| **E 探索** | `single_r3_explore` | lr 0.005→0.015 | 探索量 |
| **D DIP** | `single_r3_dip` | generator direct→sigmoid | 架構連通先驗 |
| **E+D** | `single_r3_dip_explore` | sigmoid + lr 0.015 | 加乘 |
- **判準**: worst-margin vs HFSS-call + **像素翻轉數**（探索量對照）;連通看 r_feed。
- **HFSS 預算**: 各 500 epoch。

## 3. 執行紀錄 (Run)
| 臂 | 機器 | 狀態 | 結果夾 |
|---|---|---|---|
| E `single_r3_explore` | 216 | 2026-07-02 停 @189 ep | `[Patch-single-216-0895c2] pixel_single_r3_explore` |
| D `single_r3_dip` | 37（快） | 2026-07-02 停 @101 ep | `[Patch-single-37-8dbe23] pixel_single_r3_dip` |
| E+D `single_r3_dip_explore` | 218 | 2026-07-02 停 @132 ep | `[Patch-single-218-9789f4] pixel_single_r3_dip_explore` |
- **2026-07-01 發**（不等 Round 2 到 500）：停 Round 2 釋放 216/37/218。⚠ **機器速度不均**：37 ~6分/ep（~2 天到 500）、216/218 ~18-33分/ep（~6-11 天到 500）→ 慢機的 E/E+D 短期到不了 500，先用到得了的 epoch 對比；建議把最想快看的臂放 37（此處 D=DIP headline）。機器分配可調。
- **2026-07-01 健檢**（全臂）：三臂 `sm_fit_epochs` 全程=1（dlf 每輪只訓 elite 1 epoch）、`sm_bias` +3~4.5（SM 系統性樂觀）、`trust_t` 鎖 0.05 → 「利用」端從未啟動；E 的 best@89 之後被純探索沖走 → 定位共同瓶頸 = SM 欠訓（Round 4 由來）。
- **2026-07-02 提早停**（未到 500）：瓶頸已定位、續跑三臂資訊增量有限 → 停跑釋放機器給 Round 4（修瓶頸重跑同 factorial）。

## 4. 分析 (Analyze)
2026-07-02 產（`python -m script.round_report --round 03 --runs single_r3_explore single_r3_dip single_r3_dip_explore --labels E D E+D --at 101`；對標點取三臂最小進度 101）：

| 臂 | 最佳 worst_margin | 達到 epoch |
|---|---|---|
| E | -3.63 dB | 89 |
| E+D | -5.69 dB | 8 |
| D | -5.83 dB | 8 |

- 對照 reference ②（R2：dlf+ensemble+trust、direct、lr 0.005）最佳 **-3.87 dB@430ep** → **E 用不到 ¼ 的 epoch 追平並小勝**（+0.24 dB）→ 解凍（lr↑）是真實有效的旋鈕。
- D 與 E+D best 都停在 ep8（各自的早期解 -5.83／-5.69），之後 90+/120+ epoch 再沒刷新 = 幾乎沒有效搜尋。
- **2026-07-03 更正**：原表 D 誤記 -5.69（`_resolve_run` 子字串 bug——`single_r3_dip` 被解析到 `…dip_explore` 資料夾、兩臂數字相同）；修復後 D 真值 **-5.83@8**（與 status 的 wm 欄一致，原「0.14 dB 小差」註腳即此 bug）。原「D/E+D 同 seed 同一顆解」推論一併撤回——兩臂只是同樣早停，非同一解。圖已重產。
- random best-of-N 對照仍缺：`harvest_single_random` 資料夾存在但 **0 筆**（round-01 至今未收）→ benchmark 圖無 random 線。
- 圖：`assets/round-03/E_best.png` · `D_best.png` · `E+D_best.png` · `benchmark.png`。

## 5. 結論 (Conclude)
- **E（探索/lr↑）有效且是本輪最佳**：-3.63 dB@89，¼ epoch 追平 ② → 「搜尋凍住」假設成立、lr 是真實旋鈕。
- **D（DIP/sigmoid）連通先驗成功但結果反差**：r_feed ~0.95（vs direct ~0.2）如預期，但 worst_margin 更差且 best 停在 ep8 —— sigmoid 臂的搜尋整段停滯。
- **E+D 無可見加乘**：與 D 同樣停滯。
- **⚠ factorial 判讀被共同瓶頸汙染**：三臂同吃「SM 欠訓（dlf 每輪 1 epoch）→ sm_bias +3~4.5 → trust_t 鎖 0.05 → 永不利用」（§3 健檢）→「D 臂差」**不能**歸因 DIP 本身；本輪 factorial 不算乾淨的答案。
- **學到什麼**：先修 SM 欠訓、讓 trust 有機會進入利用，再重讀探索×DIP factorial → Round 4。

## 6. 後續決策 (Next)
- → **Round 4**：三臂全上自適應 SM 訓練量（`mode:adaptive`）重跑、去 confound（[round-04](round-04-adaptive-sm.md)）。
- direct-only 探索子臂（UCB/diversity）續留待 R4 之後。
- random 對照集 `harvest_single_random` 仍 0 筆，待正式機收 random-sim（不擋 R4）。

## 7. 歸檔指向 (Archive)
- configs/README 列: `single_r3_explore` / `single_r3_dip` / `single_r3_dip_explore`
- 結果夾: 見 §3 表；圖: `assets/round-03/`
- reference: Round-2 ②
- memory: [[project_generator_hyperfeature_pivot]] [[project_litreview_direction]]
- 接棒: [round-04](round-04-adaptive-sm.md)（修 SM 欠訓瓶頸、重跑同 factorial）
