# Round 4 — 自適應 SM 訓練量（修 R3 的 SM 欠訓瓶頸）

- 狀態: proposed
- 提出 / 開跑 / 結論: 2026-07-02 / — / —
- 一句話問題: 用「held-out 的 fresh HFSS 點」自調每輪 SM 重訓 epoch 數，能否修掉 R3 的「SM 欠訓 → trust 鎖死 → plateau」，讓 E/D/E+D factorial 終於乾淨？
- 一句話結論: —（待跑）
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
| E | 216 | 2026-07-02 發 | `[Patch-single-216-…] pixel_single_r4_explore` |
| D | 37（快） | 2026-07-02 發 | `[Patch-single-37-…] pixel_single_r4_dip` |
| E+D | 218 | 2026-07-02 發 | `[Patch-single-218-…] pixel_single_r4_dip_explore` |
- 事件 / 全域變更: 進 R4 前一批工程（Round4 準備）：新增追蹤訊號（flips/stall/sm_bias/wm_per-label/sm_train_epochs/probe_*/elite_n）；實作 opt-in 自適應-SM（golden 零漂移）。發前先跑 `python -m script.status` 掃機器真相。
- 2026-07-02 發車前健檢：修控制器「低訓練量死鎖」（float target + 本輪觀測投票，模擬實證；沒修的話 R4 很可能默默退化回 dlf、整輪白跑）＋ `seed_target()` 斷點續跑續 target ＋ `elite_n` 成本訊號。詳見 `docs/discuss/decisions.md` 補記。
- ⚠ 2026-07-02 **三臂實際起跑早於修復 push**（前 ~9–15 ep 跑在修復前代碼）：E 臂 csv 已見死鎖前兆（`sm_train_epochs` 8→3 下滑、多輪 `probe_min≈probe_max`＝探測被雜訊主導）→ 各臂停跑 → 正式機 `git pull` → 原 config 重啟（RunState 斷點續跑、`seed_target` 接手續 target；修復後 target 可從低點爬回）。判讀曲線時注意 ep≤重啟點為舊碼段。

## 4. 分析 (Analyze)
（待跑完 `python -m script.round_report --round 04 --runs … --labels E D E+D`）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 若自適應有效：考慮把探測 range 自動外擴（argmin 貼 epoch_max 時）、與 TrustController 共用 gap EMA。
- 若無效：回頭檢查是否 elite 篩選/探測範圍問題，或瓶頸另有其處。

## 7. 歸檔指向 (Archive)
- configs/README 列: `single_r4_explore` / `single_r4_dip` / `single_r4_dip_explore`
- 結果夾: （待）
- memory: [[project_sm_training_redesign]] · [[project_discussion_memory]]
- 設計文件: `docs/discuss/decisions.md`「自適應 SM 訓練量」
- ONGOING 動作: 跑完把 Round 4 移出 🔵、✅ 區留一行指標
