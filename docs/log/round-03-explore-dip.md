# Round 03 — 探索 × DIP（factorial：E / D / E+D）

- **狀態**: running  <!-- 2026-07-01 發(不等 Round 2 到 500;② ~417ep 當 reference);各 500 epoch -->
- **提出 / 開跑 / 結論**: 2026-07-01 / 2026-07-01 / —
- **一句話問題**: (E) 搜尋凍住(每 epoch 才翻 ~6 像素)→ 加大步長探索能解凍+變好嗎？(D) 把 generator 帶回來(sigmoid 連通先驗)能救 S11 不共振嗎？(E+D) 兩者加乘？
- **一句話結論 (TL;DR)**: 待分析
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
| 臂 | 機器（計畫） | 狀態 | 結果夾 |
|---|---|---|---|
| E `single_r3_explore` | 216 | 2026-07-01 發 | 跑起後填 |
| D `single_r3_dip` | 37（快） | 2026-07-01 發 | 跑起後填 |
| E+D `single_r3_dip_explore` | 218 | 2026-07-01 發 | 跑起後填 |
- **2026-07-01 發**（不等 Round 2 到 500）：停 Round 2 釋放 216/37/218。⚠ **機器速度不均**：37 ~6分/ep（~2 天到 500）、216/218 ~18-33分/ep（~6-11 天到 500）→ 慢機的 E/E+D 短期到不了 500，先用到得了的 epoch 對比；建議把最想快看的臂放 37（此處 D=DIP headline）。機器分配可調。

## 4. 分析 (Analyze)
待跑。（`python -m script.round_report --round 03 --runs single_r3_explore single_r3_dip single_r3_dip_explore --labels E D E+D --at 500`）

## 5. 結論 (Conclude)
待跑。

## 6. 後續決策 (Next)
待跑。

## 7. 歸檔指向 (Archive)
- configs/README 列: `single_r3_explore` / `single_r3_dip` / `single_r3_dip_explore`
- reference: Round-2 ②
- memory: [[project_generator_hyperfeature_pivot]] [[project_litreview_direction]]
