# Round 02 — ensemble + trust（文獻治本）

- **狀態**: proposed  <!-- 待跑;benchmark(Round 1 vs random)看完 + 使用者說跑再 launch -->
- **提出 / 開跑 / 結論**: 2026-06-28 / — / —
- **一句話問題**: SM-guided 搜尋本身輸 random(sigmoid 與 direct 都輸)→ 文獻治本(不確定性 + 信任域門控 + active learning)能不能把它推過 random / 推到達標?
- **一句話結論 (TL;DR)**: 待分析
- **指向**: configs/README「Round 2」三列 · 結果夾 §3 · memory [[project_litreview_direction]] [[project_generator_hyperfeature_pivot]] · 設計文件 docs/guided_search_design.md

## 1. 假設 (Propose)
- **問題 / 假設**: Round 1 排除「SM 訓練量」;sigmoid(連通)也輸過 random → 共同病灶是「**SM-guided 梯度搜尋在 HFSS 預算內贏不過 random**」。文獻定論治本 = ensemble 不確定性門控 + 閉迴路信任(trust)+ active learning。這套早已實作但只在 under-trained dlf 上跑過(不算數)→ 在好 SM 底上乾淨測。
- **為何現在做**: 承接 Round 1 結論(訓練量非 bottleneck → 轉治本);原計畫階梯的下一級。
- **預期結果與判準**: ensemble/trust 應把 worst-margin 曲線推高、最好能過 random best-of-N(或縮小與 random 的差)。對照 Round-1 A(dlf 單 SM)、C(refit 單 SM)。
- **依據**: docs/guided_search_design.md;memory [[project_litreview_direction]]。

## 2. 實驗設計 (Design)
| 臂 | config | SM 底 | 治本內容 | 對照 |
|---|---|---|---|---|
| ① | `single_r2_ens_harvest` | dlf | ensemble(不確定性懲罰 + acquisition) | Round-1 A |
| ② | `single_r2_enstrust_harvest` | dlf | ensemble + trust(閉迴路 gap 門控) | ① + Round-1 A |
| ③ | `single_r2_refit_enstrust_harvest` | refit | ensemble + trust | Round-1 C |
- baseline 用 Round-1 A/C,不重跑控制組。rad `n_basis`=8(老師)。
- **判準**: worst-margin vs HFSS-call + random;盯 TB `sm/sm_unc`(成員分歧)、`sm/trust_t`+`sm/gap_ema`(②③ 信任控制有沒有動)。
- **HFSS 預算**: 各跑到 500 epoch。

## 3. 執行紀錄 (Run)
| 臂 | 機器 | 狀態 / 進度 | 結果夾 (NAS) |
|---|---|---|---|
| ① | — | 待跑 | — |
| ② | — | 待跑 | — |
| ③ | — | 待跑 | — |
- **事件 / 全域變更**: 跑前先停 Round-1 釋放機器。

## 4. 分析 (Analyze)
<!-- 跑完: python -m script.round_report --round 02 --runs single_r2_ens_harvest single_r2_enstrust_harvest single_r2_refit_enstrust_harvest --labels ens ens_trust refit_ens_trust --at 500 -->
待跑。

## 5. 結論 (Conclude)
待跑。

## 6. 後續決策 (Next)
- 已確認:Round 2 跑完 → **Round 03 = DIP / generator 帶回來**(測「DIP+治本 vs free+治本」;若仍碎/S11 爛則升頭號)。

## 7. 歸檔指向 (Archive)
- configs/README 列: `single_r2_ens_harvest` / `single_r2_enstrust_harvest` / `single_r2_refit_enstrust_harvest`
- memory: [[project_litreview_direction]] [[project_generator_hyperfeature_pivot]]
- 設計文件: docs/guided_search_design.md
