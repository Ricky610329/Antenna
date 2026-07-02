# 進行中實驗追蹤（ONGOING）

> 這裡是 **live 操作板**：只記「**現在在跑 / 待跑**」，保持精簡、會搬走。完整「為什麼/學到什麼」在研究日誌。
> - 研究主線時間軸（append-only 歷史）→ [../docs/log/README.md](../docs/log/README.md)
> - config 全集（不刪）→ [README.md](README.md)
> **流程**：新實驗 → `docs/log/` 開 round 檔 + 這裡加「🔵 進行中」一行指向它；跑完結論寫進 round 檔，這裡只留「✅ 已歸檔」一行指標。

最後更新：2026-07-02

**全域變更（2026-06-28）**：① 驗證預算改為**跑到 500 epoch**（約 3 天；原 250）→ Round-2 config `epochs: 500`。② **回滾機制已移除**（對 generator-free + K 候選 + 線上 SM 不合身、且原實作有 off-by-one + 覆蓋最佳檔兩個 bug）→ Round 1 的「不收斂」有它一份；探索改靠 K 候選 + SM 引導 (+ trust)。最佳 pattern 仍安全存 `patterns/`。

---

## 🔵 進行中 / 待跑

### Round 4 — 自適應 SM 訓練量（factorial，🔵 **proposed**，2026-07-02）→ 詳見 [docs/log/round-04](../docs/log/round-04-adaptive-sm.md)
| 臂 | config | = R3 同臂改什麼 | 隔離 |
|---|---|---|---|
| E 探索 | `single_r4_explore` | `mode: dlf→adaptive` | 自適應 SM 訓練量 |
| D DIP | `single_r4_dip` | `mode: dlf→adaptive` | 同上（sigmoid 臂） |
| E+D | `single_r4_dip_explore` | `mode: dlf→adaptive` | 同上（sigmoid+lr↑） |
- **由來**：R3 健檢定位共同瓶頸＝SM 欠訓（dlf 每輪 1 epoch → trust 鎖 0.05 不利用 → plateau）。R4 三臂全上**自適應訓練量**（held-out fresh 點量泛化、自調每輪 elite 重訓 epoch 數，沿途快照 member0、下一輪新點評 argmin；opt-in `mode:adaptive`、golden 零漂移）→ 移除 confound、修瓶頸。adaptive 旋鈕：`snapshots 5/epoch_min 1/epoch_max 32/ema 0.3`，ensemble 保持 5。
- **判準**：worst_margin（真目標）+ trust_t 升離 0.05 + sm_bias 降；cross-round 比 R3 同臂。**發前先跑 `python -m script.status`**、各 500 epoch。⚠ 每輪 SM 重訓量比 dlf 大，正式機量 SM 佔 HFSS 比例。
- **狀態**：config ready、工程完成（追蹤訊號 + 自適應-SM 已進 GAN）；**待使用者派工**。指令：`python train.py configs/single_r4_<E/D/E+D>.yaml`。

### Round 3 — 探索 × DIP（factorial，🔵 **running**，2026-07-01 發）→ 詳見 [docs/log/round-03](../docs/log/round-03-explore-dip.md)
| 臂 | config | = ② 改什麼 | 隔離 |
|---|---|---|---|
| E 探索 | `single_r3_explore` | lr 0.005→0.015 | 解凍/探索量 |
| D DIP | `single_r3_dip` | direct→sigmoid | 連通先驗 |
| E+D | `single_r3_dip_explore` | sigmoid + lr 0.015 | 加乘 |
- reference = 現有 ②（不重跑）。lr 是唯一對 direct&sigmoid 都通的探索旋鈕（保 factorial 乾淨）。**2026-07-01 發**（停 Round 2 釋放機器;計畫 E@216 D@37 E+D@218），各 500 epoch。⚠ 37 快、216/218 慢 ~3-6× → 慢機短期到不了 500,先用到得了的 epoch 比。判準：同時看像素翻轉數(探索量)+ worst_margin。
**問題**：sigmoid 與 direct 都輸 random → 病灶是「SM-guided 搜尋本身」。測文獻治本＝不確定性/信任門控（[[project_litreview_direction]]）。baseline 用 Round-1 的 A/C，不重跑控制組。rad `n_basis`＝8（老師）。

| 臂 | config | SM 底 | 治本內容 |
|---|---|---|---|
| ① | `single_r2_ens_harvest` | dlf | ensemble（不確定性懲罰 + acquisition） |
| ② | `single_r2_enstrust_harvest` | dlf | ensemble + trust（閉迴路 gap 門控） |
| ③ | `single_r2_refit_enstrust_harvest` | refit | ensemble + trust（最強 SM 底） |

- **待**：benchmark 看完 + 使用者說跑再 launch。指令（三台各一）：`python train.py configs/single_r2_<…>.yaml`
- **判準**：對照 Round-1 A（dlf 單 SM）、C（refit 單 SM），看 ensemble / trust 有沒有把 worst_margin 推過 random、或推到達標。

---

## 🔜 候選 / 待排（**[使用者] = 你提的**；看 benchmark + Round 2 結果再決定優先序）
- **[使用者] DIP + 探索 → 已成 Round 3（config ready）**：E(lr↑)/D(sigmoid DIP)/E+D factorial,見上方 Round 3 區塊與 [docs/log/round-03](../docs/log/round-03-explore-dip.md)。待 Round 2 判讀完後發。
  - direct-only 探索子臂（UCB `selection.uncertainty_weight`↑ / diversity↑）留待 Round 3 之後（候選式旋鈕、sigmoid 用不了,不進本輪 factorial）。
- ~~**[使用者] val-早停**~~ → **已成 Round 4**（`mode:adaptive`）：用「下一個 held-out HFSS 點」評 member0 快照、自調每輪 SM 重訓 epoch 數。見上方 Round 4 區塊與 [docs/log/round-04](../docs/log/round-04-adaptive-sm.md)。
- **[使用者] 可解釋性 / SM 歸因（AlphaFold-like）**：用 SM 做屬性分析，找「哪些像素對好 pattern 貢獻最大」→ 當設計先驗/引導。先記錄、之後測。
- **[使用者] 把「對稱」做對（下一次想試）**：現行硬 mirror（`MirrorGenerator`，**12-1-12** = 對中央 1 欄做完整左右鏡射）表現普通、可能太死。試**部分對稱**：例如 **10-5-10**（外側 10 欄左右對稱 + **中央 5 欄自由**，給饋電/中央共振區自由度），或改成**軟對稱 loss**（鼓勵而非硬鎖）。做之前先定哪種（generator 結構切法 vs loss）+ 中央自由帶寬度。動 loss 前依規矩討論。
- **[使用者] 週期 harvest 重錨（更極致 refit）**：把過往好樣本（含 harvest）週期性整批重訓 SM，讓資料越跑越多、暖啟動越來越好（現在 run 的資料不回灌中央池，這條補那塊）。
- **[使用者] 結構性先驗 → 走架構、不走 loss（主題）**：**連通** 和 **對稱** 是同一類——都是 pattern 的**結構性先驗**,適合用 **generator 架構(DIP)** 內建,而不是靠 loss 硬拉。
  - **連通**：不動 `sc loss`（**已驗證有效**）;連通交給 **DIP**（sigmoid 架構天生連通,r_feed 0.62 vs direct 0.2）→ **Round 3 D 正在測**。
  - **對稱**：10-5-10 部分對稱（見上方對稱候選）——同樣走 generator 結構切法。
  - 洞見：pattern 的結構約束（連通/對稱）架構做比 loss 做乾淨、不跟主目標搶梯度。
- **[我/發現] loss 對齊 worst_margin**：sim_loss 最低 ≠ 天線最好（Round 1 發現）；潛力大但動 loss 前討論。
- **[使用者] rad 塑形 = 弱推力（走 a；設計已定 2026-06-30）**：radiation 透過 SM rad 預測影響 pattern（beam loss 算在預測上、反傳到 logits；絕對增益歸 Gain target）。**實測現有 head 窗內 ±45° ~3.5dB**（形狀歪、非高度偏 → 改吐相對形狀沒用；是**凍 trunk 容量限制、非 n_basis**）≈ 3dB 門檻 → 不夠精確驅動 3dB 覆蓋。**(a) 走弱推力**：覆蓋項改 **worst-angle（soft-min ±45°，對齊 worst_margin）** + **低權重 nudge** + 課程化（S11/Gain OK 後升）+ rad 收尾（實際 `sm_min_loss`、`n_basis`=8）。**(b) 容量投資**（週期解凍 trunk⚠NaN / 物理 FFT）延到 radiation 變主角。詳見 [[project_radiation_pattern]]。動 loss 前討論。

---

## ✅ 已歸檔（一行指標，完整結論在 round 檔）

- **Round 01 — SM 訓練量 A/B** → [docs/log/round-01](../docs/log/round-01-sm-training-ab.md)：**訓練量非 bottleneck**(dlf −4.18≈refit −4.21 > dlf_fit −5.58、皆差 spec ~4dB)。圖 `docs/log/assets/round-01/`。
- **Round 02 — ensemble + trust 治本** → [docs/log/round-02](../docs/log/round-02-ensemble-trust.md)：**治本微幅、未決定性**(②③ trust 微贏 Round-1 ~0.3-0.5dB、① ens-only 輸、皆未收斂;trust_t 卡低)。2026-07-01 提早停(未到 500)釋放機器給 Round 3;② ~417ep 當 Round-3 reference。
