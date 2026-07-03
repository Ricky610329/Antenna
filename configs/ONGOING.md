# 進行中實驗追蹤（ONGOING）

> 這裡是 **live 操作板**：只記「**現在在跑 / 待跑**」，保持精簡、會搬走。完整「為什麼/學到什麼」在研究日誌。
> - 研究主線時間軸（append-only 歷史）→ [../docs/log/README.md](../docs/log/README.md)
> - config 全集（不刪）→ [README.md](README.md)
> **流程**：新實驗 → `docs/log/` 開 round 檔 + 這裡加「🔵 進行中」一行指向它；跑完結論寫進 round 檔，這裡只留「✅ 已歸檔」一行指標。

最後更新：2026-07-02

**全域變更（2026-06-28）**：① 驗證預算改為**跑到 500 epoch**（約 3 天；原 250）→ Round-2 config `epochs: 500`。② **回滾機制已移除**（對 generator-free + K 候選 + 線上 SM 不合身、且原實作有 off-by-one + 覆蓋最佳檔兩個 bug）→ Round 1 的「不收斂」有它一份；探索改靠 K 候選 + SM 引導 (+ trust)。最佳 pattern 仍安全存 `patterns/`。

---

## 🔵 進行中 / 待跑

### Round 5 — 滑動視窗 SM 訓練量（🔵 **running**，2026-07-03 發）→ 詳見 [docs/log/round-05](../docs/log/round-05-window-sm.md)
| 臂 | config | = R4 同臂改什麼 | 隔離 |
|---|---|---|---|
| E | `single_r5_explore` | `mode: adaptive→adaptive_window` + ensemble 5→3 | 滑動視窗訓練量（⚠兩變更） |
| D | `single_r5_dip` | 同上 | 同上（sigmoid 臂） |
| E+D | `single_r5_dip_explore` | 同上 | 同上（紀錄臂：看「撞到」能否變「開採」） |
- **由來**：R4 實錘兩件事——深度欠訓（每輪訓完 elite fit_loss 仍 7.7-10.6，學長壓 0.1）＋ adaptive 探測自鎖（target 停 3-5、曲線 80-100% 平）。**滑動視窗（Ricky 設計）**：每輪訓到視窗頂 hi、log2 階梯快照、argmin **落「上二階」**（不必貼頂——最佳點上方保留兩階冗餘）連 3 次→hi×2／落最低一階→hi÷2；起點 64、**上限 1024**（爬到頂≈學長「破千」量級）、下限 8；`replay_size 512`、`ensemble 3`（省成本）。工程完成（`mode: adaptive_window`、golden 零漂移）。
- **判準（分層）**：① fit_loss 壓到 ~1-3 → ② sm_gap/sm_bias 降、trust_t 升 → ③ worst_margin vs R4 同臂。⚠ 視窗爬到頂＝數十萬步/輪，**正式機務必盯 `time` 欄**看 SM 佔比；失控把天花板降回 256/512（一行 config）。
- **狀態**：**2026-07-03 發**（E@216、D@37、E+D@218，config 快照已驗證新版；R4 三臂已停）。各 500 epoch。

---

## 🔜 候選 / 待排（**[使用者] = 你提的**；看 benchmark + Round 2 結果再決定優先序）
- **[討論] 選擇端 known-bad 鄰域懲罰（治 R4 E ping-pong）**：acquisition 罰「採過且證實爛」的鄰域；SM 續走 elite-only（CartPole 論點：只學好的保地形、盲區問題在選擇端解）。**觸發條件：R4 結束時 trust_t 未升離 0.05 且 ping-pong（flips 雙峰）未消**；若 trust 升了它自癒、本條作廢。詳見 `docs/discuss/scratch.md`「ping-pong」塊。
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

- **Round 04 — 自適應 SM 訓練量** → [docs/log/round-04](../docs/log/round-04-adaptive-sm.md)：**E+D 破專案紀錄 -2.89@154**（探索躍遷,+2.80 vs R3）；主假設未驗證（探測自鎖 3-5ep、fit_loss 仍 8-11、trust 全鎖;E/D 輸 R3 ~0.9dB）→ R5 滑動視窗。2026-07-03 停（E@208/D@222/E+D@201）。圖 `docs/log/assets/round-04/`。
- **Round 03 — 探索 × DIP factorial** → [docs/log/round-03](../docs/log/round-03-explore-dip.md)：**E(lr↑)最佳 -3.63@89（¼ epoch 追平②）**;DIP 連通成功(r_feed~0.95)但停滯(best@8);三臂被 SM 欠訓汙染、factorial 不乾淨 → R4 修瓶頸重跑。2026-07-02 停(E@189/D@101/E+D@132)。圖 `docs/log/assets/round-03/`。
- **Round 01 — SM 訓練量 A/B** → [docs/log/round-01](../docs/log/round-01-sm-training-ab.md)：**訓練量非 bottleneck**(dlf −4.18≈refit −4.21 > dlf_fit −5.58、皆差 spec ~4dB)。圖 `docs/log/assets/round-01/`。
- **Round 02 — ensemble + trust 治本** → [docs/log/round-02](../docs/log/round-02-ensemble-trust.md)：**治本微幅、未決定性**(②③ trust 微贏 Round-1 ~0.3-0.5dB、① ens-only 輸、皆未收斂;trust_t 卡低)。2026-07-01 提早停(未到 500)釋放機器給 Round 3;② ~417ep 當 Round-3 reference。
