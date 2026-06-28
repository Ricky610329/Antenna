# 進行中實驗追蹤（ONGOING）

> 這裡只記「**現在在跑 / 待跑 / 收尾中**」的實驗，保持精簡。
> 完整 config 對照表在 [README.md](README.md)（全目錄、不刪、accumulating）。
> **流程**：新實驗 → 加進「🔵 進行中」；跑完有結論 → 搬到「✅ 已歸檔」寫一行結論（config 仍留在 README，不動）。

最後更新：2026-06-28

---

## 🔵 進行中 / 收尾中

### Round 1 — SM 訓練量 A/B（收尾中，待 benchmark）
**問題**：SM 每輪該訓多用力？輕（dlf）/ 訓菁英到飽（dlf_fit）/ 訓全部到飽（refit）。

| 臂 | config | 機器 | 狀態（2026-06-28） | 最佳 worst_margin |
|---|---|---|---|---|
| A `dlf` | `single_guided_harvest` | 216 | ~304ep，跑動中 | **−4.18 dB** |
| B `dlf_fit` | `single_guided_dlffit_harvest` | 37 | ~248ep，**simulator crash** | −5.58 dB |
| C `refit` | `single_guided_refit_harvest` | 218 | ~303ep，停/crash | **−4.21 dB** |

- **早期結論**：`dlf_fit`（B）過擬合、ep157 後 plateau ＝ 最差；`dlf`（A）≈`refit`（C）較好、還在慢慢進步。三者**皆差 spec ~4dB、未達標、搜尋未收斂**（最佳是運氣單點）。→ **SM 訓練量不是 bottleneck。**
- **待辦**：跑 `benchmark_vs_random`（vs random best-of-N @250）＝ Round 1 句點 → 跑完把結論搬「✅ 已歸檔」。

### Round 2 — ensemble + trust 治本（已產 config，待跑）
**問題**：sigmoid 與 direct 都輸 random → 病灶是「SM-guided 搜尋本身」。測文獻治本＝不確定性/信任門控（[[project_litreview_direction]]）。baseline 用 Round-1 的 A/C，不重跑控制組。rad `n_basis`＝8（老師）。

| 臂 | config | SM 底 | 治本內容 |
|---|---|---|---|
| ① | `single_r2_ens_harvest` | dlf | ensemble（不確定性懲罰 + acquisition） |
| ② | `single_r2_enstrust_harvest` | dlf | ensemble + trust（閉迴路 gap 門控） |
| ③ | `single_r2_refit_enstrust_harvest` | refit | ensemble + trust（最強 SM 底） |

- **待**：benchmark 看完 + 使用者說跑再 launch。指令（三台各一）：`python train.py configs/single_r2_<…>.yaml`
- **判準**：對照 Round-1 A（dlf 單 SM）、C（refit 單 SM），看 ensemble / trust 有沒有把 worst_margin 推過 random、或推到達標。

---

## 🔜 候選 / 待排（看 benchmark + Round 2 結果再決定優先序）
- **DIP / generator-vs-free（Round 3 候選）**：帶回 generator（sigmoid/CNN）＝架構連通先驗，對照 generator-free。**理由**：generator-free 丟掉連通先驗（r_feed 0.2 vs sigmoid 0.62）→ 很可能是 S11 不共振的主因。**何時**：Round 2（治本）之後，測「DIP + 治本 vs free + 治本」（sigmoid 單獨已輸過 random → 必須配治本一起測）。**若 Round 2 仍碎/ S11 爛 → 升為頭號。**
- **連通 sc↑**：顯式連通 loss（DIP 的替代/互補），便宜先試。
- **loss 對齊 worst_margin**：sim_loss 最低 ≠ 天線最好（Round 1 發現）；潛力大但動 loss 前先討論。
- **val-早停**：用「下一個 HFSS 點」當 validation 挑 SM 最佳 epoch、防過擬合（治本的配套）。

---

## ✅ 已歸檔（結論）

> 還沒有。Round 1 的 benchmark 出來後，第一個搬進來（含「贏不贏 random」的結論 + 最佳 pattern 圖位置）。
