# 進行中實驗追蹤（ONGOING）

> 這裡是 **live 操作板**：只記「**現在在跑 / 待跑**」，保持精簡、會搬走。完整「為什麼/學到什麼」在研究日誌。
> - 研究主線時間軸（append-only 歷史）→ [../docs/log/README.md](../docs/log/README.md)
> - config 全集（不刪）→ [README.md](README.md)
> **流程**：新實驗 → `docs/log/` 開 round 檔 + 這裡加「🔵 進行中」一行指向它；跑完結論寫進 round 檔，這裡只留「✅ 已歸檔」一行指標。

最後更新：2026-06-28

**全域變更（2026-06-28）**：① 驗證預算改為**跑到 500 epoch**（約 3 天；原 250）→ Round-2 config `epochs: 500`。② **回滾機制已移除**（對 generator-free + K 候選 + 線上 SM 不合身、且原實作有 off-by-one + 覆蓋最佳檔兩個 bug）→ Round 1 的「不收斂」有它一份；探索改靠 K 候選 + SM 引導 (+ trust)。最佳 pattern 仍安全存 `patterns/`。

---

## 🔵 進行中 / 待跑

### Round 2 — ensemble + trust 治本（已產 config，待跑）→ 詳見 [docs/log/round-02](../docs/log/round-02-ensemble-trust.md)
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
- **[使用者] DIP / generator 帶回來（Round 3）✔ 已確認：Round 2 跑完就做這個**：帶回 generator（sigmoid/CNN）＝架構連通先驗，對照 generator-free。理由：generator-free 丟掉連通先驗（r_feed 0.2 vs sigmoid 0.62）→ 可能是 S11 不共振主因。做法：測「DIP+治本 vs free+治本」（sigmoid 單獨已輸過 random → 必須配 Round-2 的治本一起測）。
- **[使用者] val-早停**：用「下一個 HFSS 點」當 validation、挑 SM 最佳「訓練 epoch」、防過擬合。`sm_gap` 是訊號(眼睛、已在記)，這是手(還沒做)。治本配套。
- **[使用者] 可解釋性 / SM 歸因（AlphaFold-like）**：用 SM 做屬性分析，找「哪些像素對好 pattern 貢獻最大」→ 當設計先驗/引導。先記錄、之後測。
- **[使用者] 把「對稱」做對（下一次想試）**：現行硬 mirror（`MirrorGenerator`，**12-1-12** = 對中央 1 欄做完整左右鏡射）表現普通、可能太死。試**部分對稱**：例如 **10-5-10**（外側 10 欄左右對稱 + **中央 5 欄自由**，給饋電/中央共振區自由度），或改成**軟對稱 loss**（鼓勵而非硬鎖）。做之前先定哪種（generator 結構切法 vs loss）+ 中央自由帶寬度。動 loss 前依規矩討論。
- **[使用者] 週期 harvest 重錨（更極致 refit）**：把過往好樣本（含 harvest）週期性整批重訓 SM，讓資料越跑越多、暖啟動越來越好（現在 run 的資料不回灌中央池，這條補那塊）。
- **[我/發現] 連通 sc↑**：顯式連通 loss（DIP 的替代/互補），便宜先試。
- **[我/發現] loss 對齊 worst_margin**：sim_loss 最低 ≠ 天線最好（Round 1 發現）；潛力大但動 loss 前討論。
- **[使用者] rad 塑形 = 弱推力（走 a；設計已定 2026-06-30）**：radiation 透過 SM rad 預測影響 pattern（beam loss 算在預測上、反傳到 logits；絕對增益歸 Gain target）。**實測現有 head 窗內 ±45° ~3.5dB**（形狀歪、非高度偏 → 改吐相對形狀沒用；是**凍 trunk 容量限制、非 n_basis**）≈ 3dB 門檻 → 不夠精確驅動 3dB 覆蓋。**(a) 走弱推力**：覆蓋項改 **worst-angle（soft-min ±45°，對齊 worst_margin）** + **低權重 nudge** + 課程化（S11/Gain OK 後升）+ rad 收尾（實際 `sm_min_loss`、`n_basis`=8）。**(b) 容量投資**（週期解凍 trunk⚠NaN / 物理 FFT）延到 radiation 變主角。詳見 [[project_radiation_pattern]]。動 loss 前討論。

---

## ✅ 已歸檔（一行指標，完整結論在 round 檔）

- **Round 01 — SM 訓練量 A/B** → [docs/log/round-01](../docs/log/round-01-sm-training-ab.md)：**訓練量非 bottleneck**(dlf −4.18≈refit −4.21 > dlf_fit −5.58、皆差 spec ~4dB)。圖 `docs/log/assets/round-01/`。(機器 216/37/218 待釋放給 Round 2。)
